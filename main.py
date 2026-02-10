
def main():
    # example_to_run = "minimal_joint"  # change this to run a different example
    example_to_run = "minimal_bone_muscle_import"  # change this to run a different example

    if example_to_run == "minimal_joint":
        from examples import example_minimal_joint
        example_minimal_joint.main()
    elif example_to_run == "minimal_couple":
        from examples import example_minimal_couple
        example_minimal_couple.main()
    elif example_to_run == "minimal_bone_muscle_import":
        from examples import example_minimal_bone_muscle_import
        example_minimal_bone_muscle_import.main()

if __name__ == "__main__":
    main()
