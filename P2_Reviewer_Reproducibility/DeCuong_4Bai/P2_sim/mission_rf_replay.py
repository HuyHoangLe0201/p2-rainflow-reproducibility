"""Forced-action replay for the deterministic NPC mission comparison."""
import numba
import numpy as np

from core import SHIFT, STATES, advance_thermal, envelope_prediction, losses, rf_push, rf_total
from viability_helpers import assess_vector


@numba.njit(cache=True)
def replay(profile, regimes, actions, lut, resistance, tau, dt, block_steps, record_rf):
    current = np.zeros(3)
    previous_state = np.zeros(3, dtype=np.int64)
    edges = np.zeros((50, 12), dtype=np.int64)
    edge_sum = np.zeros(12, dtype=np.int64)
    local = np.zeros((30, 4))
    shared = np.zeros(9)
    upper = local.copy()
    shared_upper = shared.copy()
    stack = np.full((30, 4096), 40.)
    depth = np.ones(30, dtype=np.int64)
    last = np.full(30, 40.)
    direction = np.zeros(30, dtype=np.int64)
    settled = np.zeros(30)
    ledger = np.zeros((1 + len(profile) // block_steps, 30))
    service = np.zeros((3, 4))  # steps, per-phase current SSE, actual work, reference work
    limits = np.zeros(6)  # upper peak, NP, edges, current, current error, nominal peak
    history = np.empty((2 * len(profile) + 1 if record_rf else 0, 30))
    if record_rf:
        history[0, :] = 40.
    voltage_delta, phase = 0., 0.
    failure, completed = 0, 0
    for k in range(len(profile)):
        action = actions[k]
        if action < 0 or action >= len(STATES):
            failure = 128
            break
        row = profile[k]
        state = STATES[action]
        slot = k % 50
        available = edge_sum - edges[slot]
        mask, predicted, delta, new_edges, next_phase, peak_i, peak_error = assess_vector(
            action, current, voltage_delta, phase, previous_state, available, row, upper,
            shared_upper, lut, resistance, tau, dt, .15, .2, 1.1, 150., 277.2, 90., 0.)
        if mask:
            failure = mask
            break
        cond, energy = losses(state, previous_state, current, (current + predicted) / 2, lut)
        caps, upper, shared_upper = envelope_prediction(
            upper, shared_upper, cond, energy, resistance, tau, dt, .15, .2, 1.1)
        if np.max(caps) > 150.:
            failure = 32
            break
        peak, end = advance_thermal(local, shared, cond, energy, resistance, tau, dt)
        for chip in range(30):
            if not rf_push(peak[chip], chip, stack, depth, last, direction, settled, 4.8):
                failure = 64
            if not rf_push(end[chip], chip, stack, depth, last, direction, settled, 4.8):
                failure = 64
        if failure:
            break
        if record_rf:
            history[2*k + 1] = peak
            history[2*k + 2] = end
        current, voltage_delta = predicted, delta
        previous_state = state.copy()
        edges[slot] = new_edges
        edge_sum = available + new_edges
        ipk, modulation, power_factor, fe = row
        mid_phase = phase + np.pi * fe * dt + SHIFT
        reference = ipk * np.sin(next_phase + SHIFT)
        mid_reference = ipk * np.sin(mid_phase)
        emf = (1800. * modulation / np.sqrt(3.)) * np.sin(mid_phase + np.arccos(power_factor))
        emf -= .02 * mid_reference + .00093 * ipk * 2 * np.pi * fe * np.cos(mid_phase)
        stage = regimes[k]
        service[stage, 0] += 1
        service[stage, 1] += np.sum((predicted - reference)**2) / 3
        service[stage, 2] += np.dot(predicted, emf) * dt
        service[stage, 3] += np.dot(reference, emf) * dt
        limits[0] = max(limits[0], np.max(caps))
        limits[1] = max(limits[1], abs(delta))
        limits[2] = max(limits[2], np.max(edge_sum))
        limits[3] = max(limits[3], peak_i)
        limits[4] = max(limits[4], peak_error)
        limits[5] = max(limits[5], np.max(peak), np.max(end))
        completed = k + 1
        phase = next_phase
        if completed % block_steps == 0:
            ledger[completed // block_steps, :] = settled
    snapshot = (current, previous_state, edges, edge_sum, upper, shared_upper, local, shared,
                stack, depth, last, direction, settled, np.array([voltage_delta, phase, completed]))
    return (completed, failure, service, limits, ledger, snapshot,
            rf_total(stack, depth, last, settled, 4.8), history)
