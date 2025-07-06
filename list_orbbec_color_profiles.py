import pyorbbecsdk as ob

def list_color_profiles():
    pipeline = None
    try:
        # Create a pipeline. This will automatically open the first available Orbbec device.
        pipeline = ob.Pipeline()
        print("Orbbec pipeline initialized. Attempting to list color profiles...")

        # Get the active device from the pipeline
        # The pipeline usually has a method to get the device it's currently using
        # In many pyorbbecsdk versions, you can get the device via pipeline.get_device()
        device = pipeline.get_device()

        if not device:
            print("No device found or opened by the pipeline.")
            return

        print(f"Connected to device: {device.get_usb_info().name} (Serial: {device.get_usb_info().serial_number})")

        # Get all stream profiles for the color sensor directly from the device
        # The get_stream_profiles method is typically on the device object.
        color_profiles = device.get_stream_profiles(ob.StreamType.COLOR)

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
        # Ensure the pipeline is stopped to release resources.
        if pipeline:
            pipeline.stop()
            print("Pipeline stopped.")

if __name__ == "__main__":
    list_color_profiles()
