import data_collector

text = data_collector.load_data('apple', 'keyword')
label = data_collector.load_data('AAPL', 'label')

classifier_input = data_collector.merge_price_text(text, label)
