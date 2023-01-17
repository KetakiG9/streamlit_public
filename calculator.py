import streamlit as st
st.write("###  My First Calculator")

first_number = st.number_input("Enter the First Number")
operation = st.selectbox(
"Select Operation", ("Add", "Substract", "Multiply", "Divide", "Modulus", "Raised to the Power"))
second_number = st.number_input("Enter the Second Number")
output_type = st.radio(
"Select Output Type", ("Int", "Float"))

st.write("### Output:")
# Doing required operation regarding the selected option
if output_type == "Float":
  
  if operation == "Add":
    result = (first_number+second_number)
    st.write(f"# {first_number}+{second_number} = {result}")
  elif operation == "Substract":
    result = (first_number-second_number)
    st.write(f"# {first_number}-{second_number} = {result}")
  elif operation == "Multiply":
    result = (first_number*second_number)
    st.write(f"# {first_number}*{second_number} = {result}")
  elif operation == "Divide":
    result = (first_number/second_number)
    st.write(f"# {first_number}/{second_number} = {result}")
  elif operation == "Modulus":
    result = (first_number % second_number)
    st.write(f"# {first_number}%{second_number} = {result}")
  elif operation == "Raised to the Power":
    result = (first_number**second_number)
    st.write(f"# {first_number}^{second_number} = {result}")
  else:
    st.write("Choose a valid operation")
if output_type == "Int":
  first_number = int(first_number)
  second_number = int(second_number)
  if operation == "Add":
    result = int(first_number+second_number)
    st.write(f"# {first_number}+{second_number} = {result}")
  elif operation == "Substract":
    result = int(first_number-second_number)
    st.write(f"# {first_number}-{second_number} = {result}")
  elif operation == "Multiply":
    result = int(first_number*second_number)
    st.write(f"# {first_number}*{second_number} = {result}")
  elif operation == "Divide":
    result = int(first_number/second_number)
    st.write(f"# {first_number}/{second_number} = {result}")
  elif operation == "Modulus":
    result = int(first_number % second_number)
    st.write(f"# {first_number}%{second_number} = {result}")
  elif operation == "Raised to the Power":
    result = int(first_number**second_number)
    st.write(f"# {first_number}^{second_number} = {result}")
  else:
    st.write("Choose a valid operation")