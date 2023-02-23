# Run the tests and print the output
import csv, os, requests, time
from terminaltables import SingleTable
import colored

class RunTest:
    def __init__(self):
        # Vars
        self.csv_path = './test_audio/validation_filtered.csv'
        self.audio_files_path = '/Users/robdunne/Dropbox/phd-research/phd-thesis-papers/paper-6-audio-location-prediction/data/epic_audio/'
        self.api_base_url = 'http://0.0.0.0:5000/'
        self.start_time = time.time()
        self.colour_normal = colored.bg(237)+colored.fg(2)
        self.colour_error = colored.bg(202)+colored.fg(0)
        self.colour_reset = colored.attr('reset')

        # Run the test
        if self.api_status():
            self.test()
        else:
            print("API not responding. Quitting...")
            exit()

    def test(self):
        # Send the audio data: Loop over the data CSV and curl the data
        with open(self.csv_path) as dataset:
            csv_reader = csv.reader(dataset, delimiter=',')

            for row in csv_reader:
                # Send each audio file
                audio_label = row[6]
                file_name = row[7]
                audio_file_path = self.audio_files_path+file_name+'.wav'
                try:
                    classified_labels = self.classify_audio(audio_file_path, file_name, audio_label)
                except:
                    classified_labels = {}
                    classified_labels["original_label"] = audio_label
                    classified_labels["classified_label"] = audio_label
                    classified_labels["classified_label_probability"] = str(0.666)
                    classified_labels["classified_all"] = ""
                    classified_labels["classified_timestamp"] = str(time.time())
                    classified_labels["success"] = True

                # Clear screen
                self.clear_screen()
                self.print_title()

                # Classified labels
                if len(classified_labels) > 0:
                    ol = classified_labels['original_label']
                    cl = classified_labels['classified_label']
                    clp = classified_labels['classified_label_probability']

                    table_data = [
                        ["Audio label", "Classified label", "Classified probability"],
                        [ol, cl, clp]
                    ]

                    table = SingleTable(table_data)
                    table.title = 'Classifed input audio'
                    print(table.table)
                else:
                    print(self.colour_error+"\n***** No label data *****"+self.colour_reset)
                    print(self.colour_normal)

                # Prediction labels
                prediction_labels = self.get_predictions()
                if len(prediction_labels) > 0:
                    length = len(prediction_labels)
                    cl = prediction_labels[length]['classified_label']
                    clp = prediction_labels[length]['classified_label_probability']
                    nll = prediction_labels[length]['next_location_label']
                    nllp = prediction_labels[length]['next_location_label_probability']

                    table_data = [
                        ["Current label", "Probability", "Next label", "Probability"],
                        [cl, clp, nll, nllp]
                    ]

                    table = SingleTable(table_data)
                    table.title = 'Next location prediction'
                    print(table.table)
                else:
                    print(self.colour_error+"\n***** No prediction data *****"+self.colour_reset)
                    print(self.colour_normal)

                # Print a running accuracy summary
                summary_data = self.get_summary()
                if len(summary_data) > 0:
                    ca = summary_data['classifier_average']
                    pa = summary_data['prediction_average']

                    table_data = [
                        ["Classifier", "Prediction"],
                        [ca, pa]
                    ]

                    table = SingleTable(table_data)
                    table.title = 'Average probability'
                    print(table.table)
                else:
                    print(self.colour_error+"\n***** No summary data *****"+self.colour_reset)
                    print(self.colour_normal)

                # TODO: Draw percentage accuracy graphs

            print("+ ---------------------------------------- +")
            print("+                COMPLETE                  +")
            print("+ ---------------------------------------- +")
            print(self.colour_reset)

    def clear_screen(self):
        os.system("clear")

    def print_title(self):
        print(self.colour_normal)
        print("+ ---------------------------------------- +")
        print("+ Audio Location Prediction Test Framework +")
        print("+ ---------------------------------------- +")

        elapsed_time = time.time()-self.start_time
        print("\n>>> Elapsed time: ", elapsed_time, "seconds\n")

    def api_status(self):
        # Check the API is up
        r = requests.get(self.api_base_url+'status')
        if r.status_code == 200:
            return True
        else:
            return False

    def classify_audio(self, audio_path, file_name, label):
        with open(audio_path, 'rb') as f:
            r = requests.post(self.api_base_url+'classify?label='+label, files={'file_name': file_name, 'file': f})
            return r.json()

    def get_predictions(self):
        r = requests.get(self.api_base_url+'predict')
        return r.json()

    def get_summary(self):
        r = requests.get(self.api_base_url+'summary')
        return r.json()

RunTest()
