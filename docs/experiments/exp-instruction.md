# List of Experiments
- newton solvers
    - Performance Benchmarking.
    - Stability & Controllability.

# Experiment Instruction: Stability & Controllability Test
## Goal

Evaluate the stability and controllability of the system using a real physics engine.

## Definition of Controllability

Controllability means that different activation inputs should produce distinct and proportional responses in joint angles / torques.

## Indicators of Poor Controllability

No effect: activation = 0 and activation = 1 produce identical behavior.

Binary response: only activation = 0 is static; any non-zero activation causes a sudden jump.

Overshoot: response exceeds the desired steady state.

No settling: even with activation = 0, the system takes an excessively long time to stop.

## Indicators of Good Controllability

Larger activation → larger produced torque.

Reducing activation → reduced torque.

When activation = 0, the system eventually settles to rest within a reasonable time.