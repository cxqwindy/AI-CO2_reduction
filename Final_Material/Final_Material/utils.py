import pandas as pd
import json
import numpy as np
def generate_data():
    output_csv = pd.read_excel('data/merge_out_data_final_revised.xlsx')
    output_npy = list()
    for _, one_line in output_csv.iterrows():
        print(one_line.to_json())
        one_line_dict = eval(one_line.to_json())
        data_line = {'material': one_line_dict['material'],
                     'material_type': one_line_dict['material_type'],
                     'product': one_line_dict['first_product'],
                     'product_type': one_line_dict['first_product'],
                     'method': one_line_dict['control_method'],
                     'method_type': one_line_dict['control_method_type'],
                     'label': one_line_dict['first_product_faraday_efficiency']}
        output_npy.append(json.dumps(data_line))
    print(len(output_npy))
    np.save('data/raw_data_revised.npy', np.array(output_npy))

if __name__ == '__main__':
    generate_data()