import pyorbbecsdk as ob

def list_color_profiles():
    try:
        # Create a pipeline. This will automatically open the first available Orbbec device.
        pipeline = ob.Pipeline()
        print("Orbbec pipeline initialized. Attempting to list color profiles...")

        # Get all stream profiles for the color sensor
        color_profiles = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        if not color_profiles:
            print("No color stream profiles found on the connected Orbbec device.")
            return

        print("\n--- Available Color Stream Profiles ---")
        for profile in color_profiles:
            if isinstance(profile, ob.VideoStreamProfile):
                # VideoStreamProfile has width, height, and fps
                print(f"  - Format: {profile.format()}, "
                      f"Resolution: {profile.width()}x{profile.height()}, "
                      f"FPS: {profile.fps()}")
            elif isinstance(profile, ob.StreamProfile):
                # Generic StreamProfile (less detailed, but good for completeness)
                print(f"  - Format: {profile.format()}")
            else:
                print(f"  - Unknown profile type: {type(profile)}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # It's good practice to ensure the pipeline is stopped, though it might
        # implicitly clean up on script exit.
        if 'pipeline' in locals() and pipeline:
            pipeline.stop()
            print("Pipeline stopped.")

if __name__ == "__main__":
    list_color_profiles()
