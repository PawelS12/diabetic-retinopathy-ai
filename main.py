import duplicates
import image_cropping
import folder_distributor


def main():
    image_cropping.crop_image()
    duplicates.remove_duplicates()
    folder_distributor.split_photos()


if __name__ == "__main__":
    main()
