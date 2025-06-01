import pandas as pd

# --- Configuration ---
EXCEL_FILE_PATH = 'data/raw_sales_inventory_data.xlsx' # Replace with your actual file path

# Sheet names
SALES_SHEET_NAME = 'Sales'         # Replace with the name of your sales sheet
INVENTORY_SHEET_NAME = 'Inventory' # Replace with the name of your inventory sheet

# Column names in the SALES sheet
SALES_DATE_COLUMN = 'Shipment Date'             # Replace with the actual date column name in sales sheet
SALES_FNSKU_COLUMN = 'FNSKU'           # Replace with the actual FNSKU column name in sales sheet
SALES_VALUE_COLUMN = 'Shipped Quantity'      # Replace with the column representing sales figures

# Column names in the INVENTORY sheet
INVENTORY_DATE_COLUMN = 'Date'         # Replace with the actual date column name in inventory sheet
INVENTORY_FNSKU_COLUMN = 'FNSKU'       # Replace with the actual FNSKU column name in inventory sheet
INVENTORY_VALUE_COLUMN = 'Ending Warehouse Balance'   # Replace with the column representing inventory level
# --- End Configuration ---

# --- Desired Output Column Names ---
OUTPUT_FNSKU_COL = 'FNSKU'
OUTPUT_DATE_COL = 'Date'
OUTPUT_SALES_COL = 'units_sold'
OUTPUT_INVENTORY_COL = 'inventory_level'
# --- End Desired Output Column Names ---

def load_and_prepare_data():
    """Loads, cleans, and prepares sales and inventory data from an Excel file."""
    try:
        print(f"Loading sales data from sheet: '{SALES_SHEET_NAME}'...")
        sales_df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SALES_SHEET_NAME)
        print(f"Sales data loaded successfully. Shape: {sales_df.shape}")


        print(f"\nLoading inventory data from sheet: '{INVENTORY_SHEET_NAME}'...")
        inventory_df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=INVENTORY_SHEET_NAME)
        print(f"Inventory data loaded successfully. Shape: {inventory_df.shape}")
    

    except FileNotFoundError:
        print(f"ERROR: Excel file not found at '{EXCEL_FILE_PATH}'. Please check the path.")
        return None, None
    except ValueError as e:
        print(f"ERROR: Could not find sheet names. Details: {e}")
        print(f"Please ensure '{SALES_SHEET_NAME}' and '{INVENTORY_SHEET_NAME}' are correct sheet names in your Excel file.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")
        return None, None

    # --- Sales Data Preparation ---
    try:
        sales_df.columns = sales_df.columns.str.strip()
        required_sales_cols = [SALES_DATE_COLUMN, SALES_FNSKU_COLUMN, SALES_VALUE_COLUMN]
        for col in required_sales_cols:
            if col not in sales_df.columns:
                print(f"ERROR: Sales sheet is missing required column: '{col}'. Found columns: {sales_df.columns.tolist()}")
                return None, None

        sales_df[SALES_DATE_COLUMN] = pd.to_datetime(sales_df[SALES_DATE_COLUMN])
        sales_df[SALES_FNSKU_COLUMN] = sales_df[SALES_FNSKU_COLUMN].astype(str).str.strip()
        sales_df = sales_df.rename(columns={
            SALES_DATE_COLUMN: OUTPUT_DATE_COL,
            SALES_FNSKU_COLUMN: OUTPUT_FNSKU_COL,
            SALES_VALUE_COLUMN: OUTPUT_SALES_COL
        })
        sales_df = sales_df[[OUTPUT_DATE_COL, OUTPUT_FNSKU_COL, OUTPUT_SALES_COL]]
    except Exception as e:
        print(f"Error processing sales data: {e}")
        return None, None

    # --- Inventory Data Preparation ---
    try:
        inventory_df.columns = inventory_df.columns.str.strip()
        required_inventory_cols = [INVENTORY_DATE_COLUMN, INVENTORY_FNSKU_COLUMN, INVENTORY_VALUE_COLUMN]
        for col in required_inventory_cols:
            if col not in inventory_df.columns:
                print(f"ERROR: Inventory sheet is missing required column: '{col}'. Found columns: {inventory_df.columns.tolist()}")
                return None, None

        inventory_df[INVENTORY_DATE_COLUMN] = pd.to_datetime(inventory_df[INVENTORY_DATE_COLUMN])
        inventory_df[INVENTORY_FNSKU_COLUMN] = inventory_df[INVENTORY_FNSKU_COLUMN].astype(str).str.strip()
        inventory_df = inventory_df.rename(columns={
            INVENTORY_DATE_COLUMN: OUTPUT_DATE_COL,
            INVENTORY_FNSKU_COLUMN: OUTPUT_FNSKU_COL,
            INVENTORY_VALUE_COLUMN: OUTPUT_INVENTORY_COL
        })
        inventory_df = inventory_df[[OUTPUT_DATE_COL, OUTPUT_FNSKU_COL, OUTPUT_INVENTORY_COL]]
    except Exception as e:
        print(f"Error processing inventory data: {e}")
        return None, None

    # --- Handle potential duplicates in keys before merging ---
    if sales_df.duplicated(subset=[OUTPUT_DATE_COL, OUTPUT_FNSKU_COL]).any():
        print(f"WARNING: Duplicate {OUTPUT_DATE_COL}/{OUTPUT_FNSKU_COL} combinations found in sales data. Aggregating by summing '{OUTPUT_SALES_COL}'.")
        sales_df = sales_df.groupby([OUTPUT_DATE_COL, OUTPUT_FNSKU_COL], as_index=False)[OUTPUT_SALES_COL].sum()

    if inventory_df.duplicated(subset=[OUTPUT_DATE_COL, OUTPUT_FNSKU_COL]).any():
        print(f"WARNING: Duplicate {OUTPUT_DATE_COL}/{OUTPUT_FNSKU_COL} combinations found in inventory data. Aggregating by taking the last '{OUTPUT_INVENTORY_COL}'.")
        inventory_df = inventory_df.groupby([OUTPUT_DATE_COL, OUTPUT_FNSKU_COL], as_index=False)[OUTPUT_INVENTORY_COL].last()

    return sales_df, inventory_df

def merge_and_structure_data(sales_df, inventory_df):
    """Merges sales and inventory dataframes and structures to desired output."""
    if sales_df is None or inventory_df is None:
        return None

    print("\nMerging sales and inventory data...")
    merged_df = pd.merge(sales_df, inventory_df,
                         on=[OUTPUT_DATE_COL, OUTPUT_FNSKU_COL],
                         how='outer')

    # Sort by Date and FNSKU first
    merged_df = merged_df.sort_values(by=[OUTPUT_DATE_COL, OUTPUT_FNSKU_COL])

    # Reorder columns to the desired structure: FNSKU, Date, units_sold, inventory_level
    desired_column_order = [OUTPUT_FNSKU_COL, OUTPUT_DATE_COL, OUTPUT_SALES_COL, OUTPUT_INVENTORY_COL]
    
    # Ensure all desired columns are present (some might be missing if one df was empty before merge)
    for col in desired_column_order:
        if col not in merged_df.columns:
            merged_df[col] = pd.NA # Or some other default like 0 or an empty string

    merged_df = merged_df[desired_column_order].reset_index(drop=True)

    print("Merge and structuring complete.")
    return merged_df

if __name__ == "__main__":
    sales_data, inventory_data = load_and_prepare_data()

    if sales_data is not None and inventory_data is not None:
        print("\n--- Prepared Sales Data (first 5 rows) ---")
        print(sales_data.head())
        print("\n--- Prepared Inventory Data (first 5 rows) ---")
        print(inventory_data.head())

        final_df = merge_and_structure_data(sales_data, inventory_data)

        if final_df is not None:
            print("\n--- Final Merged and Structured Data (first 20 rows) ---")
            print(final_df.head(20))

            print("\n--- Final Data Info ---")
            final_df.info()

            # Optional: Save the merged data to a new Excel or CSV file
            try:
                output_filename_csv = 'data/sales_inventory_data.csv'
                final_df.to_csv(output_filename_csv, index=False)
                print(f"\nSuccessfully saved merged data to '{output_filename_csv}'")

            except Exception as e:
                print(f"Error saving merged file: {e}")
    else:
        print("Could not proceed with merging due to errors in loading/preparing data.")