# # import streamlit as st
# # import pandas as pd
# # import matplotlib.pyplot as plt

# # # st.set_page_config(layout="wide", page_title="Simple Streamlit App")
# # # st.title("Hello, Streamlit!")
# # # st.write("This is a simple Streamlit app.")
# # # st.write({"key": ["value"]})

# # st.set_page_config(page_title="Dashboard", layout="wide")
# # st.title("Simple Data Dashboard")

# # uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "xls", "csv"])

# # # if uploaded_file is not None:
# # #     df = pd.read_csv(uploaded_file)
# # #     st.write("Data Preview:")
# # #     st.dataframe(df.head())

# # #     st.write("Statistical Summary:")
# # #     st.write(df.describe())

# # #     numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
# # #     if numeric_columns:
# # #         selected_column = st.selectbox("Select a numeric column to plot", numeric_columns)
# # #         fig, ax = plt.subplots()
# # #         ax.hist(df[selected_column].dropna(), bins=30, color='skyblue', edgecolor='black')
# # #         ax.set_title(f'Histogram of {selected_column}')
# # #         ax.set_xlabel(selected_column)
# # #         ax.set_ylabel('Frequency')
# # #         st.pyplot(fig)
# # #     else:
# # #         st.write("No numeric columns available for plotting.")

# # if uploaded_file is not None:
# #     # st.write("File uploaded successfully!")
# #     df = pd.read_csv(uploaded_file)
    
# #     st.subheader("Data Preview")
# #     st.write(df.head())
    
# #     st.subheader("Statistical Summary")
# #     st.write(df.describe())
    
# #     st.subheader("Filter Data")
# #     columns = df.columns.tolist()
# #     selected_column = st.selectbox("Select a column to filter by:", columns)
# #     unique_values = df[selected_column].unique()
# #     selected_value = st.selectbox("Select a value to filter:", unique_values)
    
# #     filtered_df = df[df[selected_column] == selected_value]
# #     st.write(filtered_df)
    
# #     st.subheader("Data Visualization")
# #     x_column = st.selectbox("Select X-axis column:", columns)
# #     y_column = st.selectbox("Select Y-axis column:", columns)
    
# #     if st.button("Generate Plot"):
# #         # fig, ax = plt.subplots()
# #         # ax.scatter(df[x_column], df[y_column], color='skyblue', edgecolor='black')
# #         # ax.set_title(f'Scatter Plot of {y_column} vs {x_column}')
# #         # ax.set_xlabel(x_column)
# #         # ax.set_ylabel(y_column)
# #         # st.pyplot(fig)
# #         st.line_chart(filtered_df.set_index(x_column)[y_column])
# #     else:
# #         st.write("Select columns and click 'Generate Plot' to see the visualization.")    # st.write("File uploaded successfully!")
        


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt

# st.set_page_config(page_title="Dashboard", layout="wide")
# st.title("📊 Simple Data Dashboard (CSV & Excel Support)")

# # File uploader for CSV and Excel
# uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "xls", "csv"])

# if uploaded_file is not None:
#     file_name = uploaded_file.name.lower()

#     # --- Read file based on extension ---
#     if file_name.endswith(".csv"):
#         df = pd.read_csv(uploaded_file)
#         st.success(f"✅ CSV file '{uploaded_file.name}' uploaded successfully.")

#     elif file_name.endswith((".xlsx", ".xls")):
#         # Handle Excel with multi-sheet support
#         excel_file = pd.ExcelFile(uploaded_file)
#         sheet_name = st.selectbox("Select a sheet to load:", excel_file.sheet_names)
#         df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
#         st.success(f"✅ Excel file '{uploaded_file.name}' uploaded successfully.")

#     else:
#         st.error("❌ Unsupported file format.")
#         df = None

#     # --- Continue only if df is loaded ---
#     if df is not None:
#         st.subheader("Data Preview")
#         st.write(df)

#         st.subheader("Statistical Summary")
#         st.write(df.describe())

#         # --- Filter Data ---
#         st.subheader("Filter Data")
#         columns = df.columns.tolist()
#         selected_column = st.selectbox("Select a column to filter by:", columns)
#         # unique_values = df[selected_column].dropna().unique()
#         unique_values = df[selected_column].unique()
#         selected_value = st.selectbox("Select a value to filter:", unique_values)

#         filtered_df = df[df[selected_column] == selected_value]
#         st.write(filtered_df)

#         # --- Data Visualization ---
#         st.subheader("Data Visualization")
#         x_column = st.selectbox("Select X-axis column:", columns)
#         y_column = st.selectbox("Select Y-axis column:", columns)

#         if st.button("Generate Plot"):
#             try:
#                 st.line_chart(filtered_df.set_index(x_column)[y_column])
#             except Exception as e:
#                 st.error(f"⚠️ Error generating plot: {e}")
#         else:
#             st.write("Select columns and click 'Generate Plot' to see the visualization.")


import streamlit as st # pyright: ignore[reportMissingImports]
import pandas as pd

st.set_page_config(page_title="Attendance Filter", layout="wide")
st.title("📅 Attendance Data Filter")

# Upload file (Excel or CSV)
uploaded_file = st.file_uploader("Upload Attendance File", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    file_name = uploaded_file.name.lower()

    # Read Excel or CSV
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        # Excel file — handle multiple sheets if any
        excel_file = pd.ExcelFile(uploaded_file)
        if len(excel_file.sheet_names) > 1:
            sheet_name = st.selectbox("Select a sheet:", excel_file.sheet_names)
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        else:
            df = pd.read_excel(uploaded_file)

    st.subheader("📋 Data Preview")
    st.dataframe(df)

    # Ensure Date column is in datetime format (if present)
    if 'Date/Time' in df.columns:
        df['Date/Time'] = pd.to_datetime(df['Date/Time'], errors='coerce')

    # --- Filter by Name ---
    if 'Name' in df.columns:
        names = df['Name'].dropna().unique()
        selected_name = st.selectbox("Select Name:", names)
    else:
        st.error("❌ 'Name' column not found in the file.")
        st.stop()

    # --- Filter by Date ---
    if 'Date/Time' in df.columns:
        min_date = df['Date/Time'].min()
        max_date = df['Date/Time'].max()
        selected_date = st.date_input("Select Date:", value=min_date, min_value=min_date, max_value=max_date)
    else:
        st.error("❌ 'Date' column not found in the file.")
        st.stop()

    # --- Apply filters ---
    filtered_df = df[(df['Name'] == selected_name) & (df['Date/Time'].dt.date == selected_date)]

    st.subheader("✅ Filtered Result")
    st.dataframe(filtered_df)

    if filtered_df.empty:
        st.warning("⚠️ No records found for the selected Name and Date.")


# import streamlit as st
# import pandas as pd

# st.set_page_config(page_title="Attendance Log Filter", layout="wide")
# st.title("📄 Attendance Log Filter (TXT File)")

# uploaded_file = st.file_uploader("Upload ALOG_001.txt file", type=["txt", "csv"])

# if uploaded_file is not None:
#     # Read the TXT file as tab/space delimited
#     df = pd.read_csv(uploaded_file, sep=r"\s+")
#     st.success("✅ File loaded successfully!")

#     st.subheader("📋 Data Preview")
#     st.dataframe(df.head())

#     # Convert DateTime to datetime object
#     if 'DateTime' in df.columns:
#         df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')

#         # Extract just the date part
#         df['Date'] = df['DateTime'].dt.date
#     else:
#         st.error("❌ 'DateTime' column not found.")
#         st.stop()

#     # --- Filter by Name ---
#     if 'Name' in df.columns:
#         names = df['Name'].dropna().unique()
#         selected_name = st.selectbox("Select Name:", sorted(names))
#     else:
#         st.error("❌ 'Name' column not found.")
#         st.stop()

#     # --- Filter by Date ---
#     min_date = df['Date'].min()
#     max_date = df['Date'].max()
#     selected_date = st.date_input("Select Date:", value=min_date, min_value=min_date, max_value=max_date)

#     # --- Apply filters ---
#     filtered_df = df[(df['Name'] == selected_name) & (df['Date'] == selected_date)]

#     st.subheader("✅ Filtered Records")
#     st.dataframe(filtered_df)

#     if filtered_df.empty:
#         st.warning("⚠️ No records found for the selected Name and Date.")


# import streamlit as st
# import pandas as pd

# st.set_page_config(page_title="Attendance Log Filter", layout="wide")
# st.title("📄 Attendance Log Filter (TXT File with UTF-16)")

# uploaded_file = st.file_uploader("Upload ALOG_001.txt file", type=["txt", "csv"])

# if uploaded_file is not None:
#     try:
#         # Read the TXT file with UTF-16 encoding
#         df = pd.read_csv(uploaded_file, sep=r"\s+", encoding="utf-16")
#         st.success("✅ File loaded successfully!")
#     except UnicodeDecodeError:
#         # Fallback in case it's UTF-16-LE
#         df = pd.read_csv(uploaded_file, sep=r"\s+", encoding="utf-16-le")
#         st.info("ℹ️ Fallback to UTF-16-LE encoding worked.")

#     st.subheader("📋 Data Preview")
#     st.dataframe(df.head())

#     # Convert DateTime column to datetime
#     if 'DateTime' in df.columns:
#         df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
#         df['Date'] = df['DateTime'].dt.date
#     else:
#         st.error("❌ 'DateTime' column not found.")
#         st.stop()

#     # Filter by Name
#     if 'Name' in df.columns:
#         names = df['Name'].dropna().unique()
#         selected_name = st.selectbox("Select Name:", sorted(names))
#     else:
#         st.error("❌ 'Name' column not found.")
#         st.stop()

#     # Filter by Date
#     min_date = df['Date'].min()
#     max_date = df['Date'].max()
#     selected_date = st.date_input("Select Date:", value=min_date, min_value=min_date, max_value=max_date)

#     # Apply filters
#     filtered_df = df[(df['Name'] == selected_name) & (df['Date'] == selected_date)]

#     st.subheader("✅ Filtered Records")
#     st.dataframe(filtered_df)

#     if filtered_df.empty:
#         st.warning("⚠️ No records found for the selected Name and Date.")


# import streamlit as st
# import pandas as pd

# st.set_page_config(page_title="Attendance Log Filter", layout="wide")
# st.title("📄 Attendance Log Filter (TXT File with Encoding & Skipping Bad Lines)")

# uploaded_file = st.file_uploader("Upload ALOG_001.txt file", type=["txt", "csv"])

# if uploaded_file is not None:
#     # Read the TXT file using utf-16, skip bad lines, and use python engine
#     df = pd.read_csv(
#         uploaded_file,
#         sep=r"\s+",
#         encoding="utf-16",
#         engine='python',
#         on_bad_lines='skip'
#     )

#     st.success("✅ File loaded successfully!")
#     st.subheader("📋 Data Preview")
#     st.dataframe(df.head())

#     # Convert DateTime column to datetime
#     if 'DateTime' in df.columns:
#         df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
#         df['Date'] = df['DateTime'].dt.date
#     else:
#         st.error("❌ 'DateTime' column not found.")
#         st.stop()

#     # Filter by Name
#     if 'Name' in df.columns:
#         names = df['Name'].dropna().unique()
#         selected_name = st.selectbox("Select Name:", sorted(names))
#     else:
#         st.error("❌ 'Name' column not found.")
#         st.stop()

#     # Filter by Date
#     min_date = df['Date'].min()
#     max_date = df['Date'].max()
#     selected_date = st.date_input("Select Date:", value=min_date, min_value=min_date, max_value=max_date)

#     # Apply filters
#     filtered_df = df[(df['Name'] == selected_name) & (df['Date'] == selected_date)]

#     st.subheader("✅ Filtered Records")
#     st.dataframe(filtered_df)

#     if filtered_df.empty:
#         st.warning("⚠️ No records found for the selected Name and Date.")


