import time
import duplicates
import image_cropping
import folder_distributor
import sets_division


def main():
    image_cropping.crop_image()
    print("Waiting")
    print("Waiting")
    print("Waiting")
    time.sleep(5)
    duplicates.remove_duplicates()
    folder_distributor.split_photos()
    sets_division.split_into_sets()


if __name__ == "__main__":
    main()
