import numpy as np
import os

def print_dict(dictionary, title ='Name of Network Here', decimal_precision=4):
    """
    Prints the information within a generated dictionary in a directly readable way
    :param dictionary: Dictionary or Dictionary of dictionaries with the descriptors to print
    :param title: Title of the print, the name of the network
    :param decimal_precision: The required precision for floating point numbers
    :return:
    None
    """
    print('-' * 50)
    print(title+':')

    if type(dictionary[list(dictionary.keys())[0]]) is not dict:
        dictionary = {'Network descriptors': dictionary}
    #Print all the descriptors for each key (usually each node)
    for node in dictionary.keys():
        print('\t'+node+":")
        for key, value in dictionary[node].items():
            if type(value) is not int:
                value = np.round(value, decimals=decimal_precision)
            print("\t\t"+key+": "+str(value))

    print('-'*50+'\n')

def print_nodes_csv(dictionary, precision = 8, title='Example.csv', delimiter=','):
    """
    Save the information of the print in a CSV file.
    :param dictionary: Dictionary or Dictionary of dictionaries with the descriptors to save.
    :param precision: The required precision for floating point numbers.
    :param title: Title of the csv.
    :param delimiter: The delimiter to use in the CSV
    :return:
    None
    """
    #Takes a row of the dictionary as example for generating the structure
    example = dictionary[list(dictionary.keys())[0]]

    #Give the dictionaries values in a matrix way with format: [[key (node_name), value1, value2...], ...]
    data = [[key]+[value for _, value in node.items()]for key, node in dictionary.items()]

    # Defines a lambda function in order to generate the data format needed for print the csv correctly
    # (Example: (integer, 8 decimals floating point, integer...))
    str_dtype_of = lambda dtype: '%i' if type(dtype) is int else '%1.' + str(precision) + 'e'
    #Generates the demanded format
    fmt = ['%s']+[str_dtype_of(value) for value in example.values()]

    #Save it in a CSV in the results directory.
    np.savetxt(fname=os.path.join('results', title), X=np.array(data, dtype=np.object), fmt=fmt,
                delimiter=delimiter, header="'Node Name', "+str(list(example.keys()))[1:-1])

    print(title+' saved')

def print_latex_table(dictionaries, column_names, decimals = 4, measuring='Descriptor/Network'):
    """
    Prints the information within a generated dictionary in a latex formatted table
    :param dictionaries: List of dictionaries with the descriptors to print
    :param column_names: Names of the columns
    :param decimal: The required precision for floating point numbers
    :param measuring: Title of the 0,0 row of the table
    :return:
    None
    """
    #Opens the table
    print('\\begin{table}[H] \n '
          '\\begin{center} \n '
          '\\begin{tabular}{'+('|l'*(len(dictionaries)+1))+'|} \n'
          '\hline\n'
          '\multicolumn{'+str(len(dictionaries)+1)+'}{|c|}{Network Descriptors} \\\\\n'
          '\hline \n'
          '\hline')

    #Prints the header of the table with all column names
    header = measuring
    for name in column_names:
        header+=' & '+ name
    header += '\\\\ \n \hline'
    print(header)

    #Prints the content of the table
    content = ''
    for key in dictionaries[0].keys():
        content+=key+' '
        for dictionary in dictionaries:
            value = dictionary[key] if type(dictionary[key]) is int else np.round(dictionary[key],decimals)
            content += ' & '+str(value)
        content += '\\\\ \hline \n'
    print(content)

    #Close the table
    print('\end{tabular} \n '
          '\caption{Network Descriptors}\n'
          '\label{table:NetDescriptors}\n'
          '\end{center}\n'
          '\end{table}')