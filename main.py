import cropping
import duplicates
import splitter
import model


if __name__ == "__main__":
    cropping.crop_image()
    duplicates.remove_duplicates()
    splitter.split_images()
    model.train()
    model.test_and_metrics()
