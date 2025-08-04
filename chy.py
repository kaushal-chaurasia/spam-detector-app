import streamlit as st
st.title("chai poll taste")
col1, col2=st.columns(2)
with col1 :
    st.header("Masala chai")
    st.image("https://media-assets.swiggy.com/swiggy/image/upload/f_auto,q_auto,fl_lossy/bfys1nvjblor8p4jd63f",width=500)
    vote1=st.button("Vote for Masala Chai")
with col2:
    st.header("Adhrakh chai")
    st.image("https://domf5oio6qrcr.cloudfront.net/medialibrary/8468/Tea.jpg",width=400)
    vote2=st.button("Vote for Adhrakh Chai")

if vote1:
    st.success("You voted for Masala Chai!")
elif vote2:
    st.success("You voted for Adhrakh Chai!")

name=st.sidebar.text_input("Enter your name:")
tea=st.sidebar.selectbox("Select your favorite tea:", ["Masala Chai", "Adhrakh Chai", "Lemon Chai", "Kesar Chai"])
if st.sidebar.button("Submit"): 
    st.sidebar.success(f"Thank you {name}! You selected {tea}.")
with st.expander("Show the chai making Instructions"):
    st.write("""To make a perfect cup of chai, follow these steps:
1. Boil water in a pot.
2. Add tea leaves and spices.
3. Pour in milk and simmer.
4. Strain and serve hot.
""")
st.markdown("### Chai Making Instructions")
st.markdown('>Blockquote: To make a perfect cup of chai, follow these steps:\n>1. Boil water in a pot.\n>2. Add tea leaves and spices.\n>3. Pour in milk and simmer.\n>4. Strain and serve hot. \n>Enjoy your chai! \n>Happy sipping!       ')
    