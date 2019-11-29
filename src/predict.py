import data_import
import train
from datetime import datetime

def main():
    ## load test data set and do feature engineering
    test = data_import.load_test()
    train.feature_eng(test)

    ## load classifier for the booking_bool
    print("Loading the Booking classifier..")
    tstart = datetime.now()
    classifier = data_import.load_model(True)
    print("Time used:" + str(datetime.now() - tstart) + "\n")

    ## predict the booking_bool
    print("Making predictions on the booking_bool..")
    tstart = datetime.now()
    book_feature_names = train.get_features(test)
    book_X =  test[book_feature_names].values
    book_Y_pred = classifier.predict_proba(book_X)[:,1]
    book_Y_pred = list(-1.0 * book_Y_pred)
    print("Time used:" + str(datetime.now() - tstart) + "\n")


    ## load classifier for the click_bool
    print("Loading the Click classifier..")
    tstart = datetime.now()
    classifier = data_import.load_model(False)
    print("Time used:" + str(datetime.now() - tstart) + "\n")

    ## predict the click_bool
    print("Making predictions on the click_bool..")
    tstart = datetime.now()
    click_feature_names = train.get_features(test)
    click_X =  test[click_feature_names].values
    click_Y_pred = classifier.predict_proba(click_X)[:,1]
    click_Y_pred = list(-1.0 * click_Y_pred)
    print("Time used:" + str(datetime.now() - tstart) + "\n")

    ## Making results where 3rd column is the score based on likelihood of click and booking
    results = zip(test["srch_id"], test["prop_id"], 4 * book_Y_pred + click_Y_pred)

    print("Writing predictions to file..")
    tstart = datetime.now()
    data_import.write_submission(results)
    print("Time used:" + str(datetime.now() - tstart) + "\n")


if __name__=="__main__":
    main()
