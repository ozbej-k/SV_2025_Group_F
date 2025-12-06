import pandas as pd
import os
import argparse
import sys

def main():
    # --- Command Line Argument Parsing ---
    parser = argparse.ArgumentParser(description='Reshape fish tracking data from long to wide format.')
    parser.add_argument('input_path', type=str, help='Path to the input CSV file')
    
    # Parse arguments
    args = parser.parse_args()
    input_csv_path = args.input_path

    # Check if file exists
    if not os.path.exists(input_csv_path):
        print(f"Error: The file '{input_csv_path}' was not found.")
        sys.exit(1)

    print(f"Processing file: {input_csv_path}...")

    df = pd.read_csv(input_csv_path)
    
    # Create variable names dynamically based on fish_id
    df['variable_name_x'] = 'x_' + df['fish_id'].astype(str)
    df['variable_name_y'] = 'y_' + df['fish_id'].astype(str)

    df_pivot_x = df.pivot(index='time', columns='variable_name_x', values='x')
    df_pivot_y = df.pivot(index='time', columns='variable_name_y', values='y')

    df_pivot_x.columns.name = None
    df_pivot_y.columns.name = None

    df_wide = pd.merge(df_pivot_x, df_pivot_y, left_index=True, right_index=True, how='outer')

    # Get all unique fish IDs present in the data to handle any number of fish
    if 'fish_id' in df.columns:
        unique_fish_ids = sorted(df['fish_id'].unique())
    else:
        print("Error: Column 'fish_id' not found in the input file.")
        sys.exit(1)

    sorted_columns = []
    for fish_id in unique_fish_ids:
        # Reconstruct the column names in the desired order: x_ID, y_ID
        sorted_columns.append(f'x_{fish_id}')
        sorted_columns.append(f'y_{fish_id}')

    # Select only the columns that exist in the dataframe
    present_columns = [col for col in sorted_columns if col in df_wide.columns]
    df_wide = df_wide[present_columns]

    df_wide = df_wide.reset_index()

    file_root, file_ext = os.path.splitext(input_csv_path)
    # Create new filename with _wide appended
    output_filename = f"{file_root}_wide{file_ext}"


    df_wide.to_csv(output_filename, index=False, float_format='%.3f')
    print(f"Data reshaped and saved to: {output_filename}")

if __name__ == "__main__":
    main()