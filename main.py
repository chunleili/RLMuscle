
def main():
    # example_to_run = "minimal_joint"  # change this to run a different example
    example_to_run = "minimal_usd_import"  # change this to run a different example

    if example_to_run == "minimal_joint":
        from examples import example_minimal_joint
        example_minimal_joint.main()
    elif example_to_run == "minimal_bone_muscle_import":
        from examples import example_minimal_bone_muscle_import
        example_minimal_bone_muscle_import.main()
    elif example_to_run == "minimal_usd_import":
        from examples import example_minimal_usd_import
        example_minimal_usd_import.main()

if __name__ == "__main__":
    main()
