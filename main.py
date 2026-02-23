import os
from dotenv import load_dotenv

def main():
    load_dotenv()
    example_to_run = os.environ.get("RUN")
    print(f"Running example: {example_to_run}")

    if example_to_run == "minimal_joint":
        from examples import example_minimal_joint
        example_minimal_joint.main()
    elif example_to_run == "minimal_bone_muscle_import":
        from examples import example_minimal_bone_muscle_import
        example_minimal_bone_muscle_import.main()
    elif example_to_run == "usd_io":
        from examples import example_usd_io
        example_usd_io.main()
    elif example_to_run == "example_dynamics":
        from examples import example_dynamics
        example_dynamics.main()
    elif example_to_run == "example_couple":
        from examples import example_couple
        example_couple.main()
    elif example_to_run == "example_muscle_warp":
        from examples import example_muscle_warp
        example_muscle_warp.main()
    elif example_to_run == "taichi_muscle":
        from VMuscle import muscle
        muscle.main()

if __name__ == "__main__":
    main()
