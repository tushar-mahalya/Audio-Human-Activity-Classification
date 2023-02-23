"""
    Markov Model from EPIC KITCHENS 100 labels
"""
import csv, os
import pandas as pd

class MM:
    training_data = '../data/epic_formatted/train_test.csv'
    validation_data = '../data/epic_formatted/validation.csv'

    training_data_filtered = '../data/epic_formatted/train_test_filtered.csv'
    validation_data_filtered = '../data/epic_formatted/validation_filtered.csv'
    audio_validation_data = '../test-framework/classifier-results.csv'

    transition_matrix = [[]]
    trainsition_matrix_index = []

    def __init__(self):
        self.build_transition_matrix()
        self.predict()

    def load_data(self, dataset):
        # Return a list of events from the Kitchen dataset CSV
        df = pd.read_csv(dataset, sep=',')
        return df['noun'].tolist()

    def build_transition_matrix(self):
        #transitions = ['A', 'B', 'B', 'C', 'B', 'A', 'D', 'D', 'A', 'B', 'A', 'D']
        transitions = self.load_data(self.training_data_filtered)

        df = pd.DataFrame(transitions)

        # Create a new column with data shifted one space
        df['shift'] = df[0].shift(-1)

        # Add a count column (for group by function)
        df['count'] = 1

        # Groupby and then unstack, fill the zeros
        transition_matrix = df.groupby([0, 'shift']).count().unstack().fillna(0)
        #print(transition_matrix)

        # Normalise by occurences and save values to get transition matrix
        #transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0).values
        self.transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0).values
        self.transition_matrix_index = list(map(lambda x: x[1], transition_matrix.iloc[0].keys().tolist()))

    def predict(self):
        # Use the transition matrix to look up the next likely event
        # Read the validation dataset into a list
        saved_results = './results.csv'
        if os.path.isfile(saved_results):
            os.remove(saved_results)
        events = self.load_data(self.audio_validation_data)
        results = pd.DataFrame(columns=['event', 'predicted_event', 'probability'])

        # Iterate over each event
        for event in events:
            # Look up the highest probability on the transition_matrix
            if event not in self.transition_matrix_index:
                continue

            event_index = self.transition_matrix_index.index(event)
            event_row = self.transition_matrix[event_index]

            predicted_probability = max(event_row)
            predicted_event = self.transition_matrix_index[list(event_row).index(predicted_probability)]

            # Ignore cyclic predictions
            #if event == predicted_event:
            #    predicted_probability = sorted(list(set(event_row)))[-2]
            #    predicted_event = self.transition_matrix_index[list(event_row).index(predicted_probability)]

            print('Current event: ', event)
            print('Predicted event: ', predicted_event)
            print('Prediction probability: ', predicted_probability)
            #exit()

            # Add to the results dataframe
            results.loc[len(results)] = [event, predicted_event, predicted_probability]

        # Calculate total accuracy
        correct = results[results['event'] == results['predicted_event'].shift(-1)]
        print(len(results))
        print(len(correct))
        print('Accuracy', (len(correct)/len(results))*100, '%')

        # Write results to file
        results.to_csv(saved_results, encoding='utf-8')

MM()
