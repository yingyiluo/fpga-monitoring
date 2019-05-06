import re
import os
import json
import sys

def parse(input_file_name, output_file_name):
    input_file = open(input_file_name, 'r')
    print input_file_name
    output_file = open(output_file_name, 'a+')
    lines = input_file.readlines()
    pattern_data = re.compile("TIME\\s+([0-9]+):  ([0-9]+) :(-[0-9]+)")
    for line in lines:
        #print "in for", line
        match_data = pattern_data.match(line)
        if match_data:
            #print "matched"
            tuple_matched = match_data.groups()
            dictionary = {'seq': long(tuple_matched[0]), 'time': long(tuple_matched[1])}
            json.dump(dictionary, output_file)
            output_file.write('\n')

parse("nounroll.log", "nounroll.json")
