"""
    Format EPIC KITCHENS 100 labels into a simpler format
    https://github.com/epic-kitchens/epic-kitchens-100-annotations
"""
import csv, os, subprocess
from urllib.request import urlretrieve
import pandas as pd

class FormatLabels:
    def __init__(self):
        self.train_test_path_input = './epic/EPIC_100_train.csv'
        self.validation_path_input = './epic/EPIC_100_validation.csv'

        self.train_test_path_output = './epic_formatted/train_test.csv'
        self.validation_path_output = './epic_formatted/validation.csv'

        self.train_test_path_filtered_output = './epic_formatted/train_test_filtered.csv'
        self.validation_path_filtered_output = './epic_formatted/validation_filtered.csv'

        #self.format_labels(self.train_test_path_input, self.train_test_path_output)
        #self.format_labels(self.validation_path_input, self.validation_path_output)
        #self.get_unique_values()

        self.format_labels_filtered(self.train_test_path_input, self.train_test_path_filtered_output, 'train')
        self.format_labels_filtered(self.validation_path_input, self.validation_path_filtered_output, 'test')

        self.reorder_by_timestamp(self.train_test_path_filtered_output)
        self.reorder_by_timestamp(self.validation_path_filtered_output)
        '''
        self.epic_audio(self.train_test_path_filtered_output, 'train')
        self.epic_audio(self.validation_path_filtered_output, 'test')
        '''

    def format_labels(self, input_file, output_file):
        # Read the CSV file
        print("Processing ", input_file, output_file)

        # Clean up and previous data
        # Loop through it and keep only the parts we need
        with open(input_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            # Write to a new CSV
            with open(output_file, mode='w') as write_file:
                writer = csv.writer(write_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                for row in csv_reader:
                    # line_count, narration_id, narration_timestamp, all_nouns
                    new_data = [line_count, row[0], row[3], row[11]]
                    writer.writerow(new_data)
                    line_count += 1

            print("Written ", line_count, "rows")

    def format_labels_filtered(self, input_file, output_file, type):
        # Read the CSV file
        print("Processing ", input_file, output_file)

        # Clean up and previous data
        # Loop through it and keep only the parts we need
        with open(input_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            # Write to a new CSV
            with open(output_file, mode='w') as write_file:
                writer = csv.writer(write_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                for row in csv_reader:
                    if row[11] in ['door', 'light', 'plate', 'noun']:
                        audio_file_name = row[2]+'_'+row[4]+'_'+row[5]+'.wav'
                        audio_file_name = audio_file_name.replace(":", "_")

                        if line_count == 0:
                            audio_file_name = 'audio_file_name'

                        # line_count, video_id, narration_timestamp, all_nouns
                        new_data = [line_count, row[2], row[3], row[4], row[5], row[11], audio_file_name]
                        writer.writerow(new_data)
                        line_count += 1

            print("Written ", line_count, "rows")

    def get_unique_values(self):
        unique_values = {}
        with open(self.train_test_path_output) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[3] in unique_values:
                    unique_values[row[3]] = unique_values[row[3]]+1
                else:
                    unique_values[row[3]] = 1

        with open('./states_count.csv', mode='w') as write_file:
            writer = csv.writer(write_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for key, value in unique_values.items():
                # line_count, narration_id, narration_timestamp, all_nouns
                new_data = [key, value]
                writer.writerow(new_data)

    def epic_audio(self, input_file, type):
        # Download and clip the audio
        with open(input_file, 'r') as read_obj:
            index = 1
            csv_reader = csv.reader(read_obj)
            header = next(csv_reader)
            # Check file as empty
            if header != None:
                # Iterate over each row after the header in the csv
                for row in csv_reader:
                    file_name = row[2]
                    start = row[4]
                    stop = row[5]
                    audio_file_name = row[7]

                    print('Processing row ', index, '...')
                    if not os.path.isfile('./epic_audio/'+audio_file_name):
                        self.fetch_clip_audio(file_name, audio_file_name, start, stop, type)

                    index = index+1

    def fetch_clip_audio(self, file_name, audio_file_name, start, stop, type):
        # Build the URL in format: https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/{P01}/videos/{filename}.mp4
        file_parts = file_name.split('_')
        if(len(file_parts[1]) == 2):
            url = 'https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/'+type+'/'+str(file_parts[0])+'/'+file_name+'.MP4'
        else:
            url = 'https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/'+str(file_parts[0])+'/videos/'+file_name+'.MP4'

        # Extract audio from the video
        # Save to audio disk as wav
        print('Processing audio for ', audio_file_name)
        command = "ffmpeg -y -i "+url+" -f wav -ss "+start+" -to "+stop+" -ab 160k -ac 1 -ar 44100 -vn ./epic_audio/"+audio_file_name
        subprocess.call(command, shell=True)

    def reorder_by_timestamp(self, file_name):
        # Open with pandas
        df = pd.read_csv(file_name)

        # Reorder
        df = df.sort_values(by='narration_timestamp',ascending=True)

        # Write file
        df.to_csv(file_name)

FormatLabels()
