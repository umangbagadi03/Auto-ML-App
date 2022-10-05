

# Main Panel

# Displays the dataset
st.subheader("1. Dataset")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown("1.1 Glimpse of Dataset")
    st.write(df)
    build_model(df)

else:
    st.info("Waiting for CSV file to be uploaded")
    if st.button('Press to use Example Dataset'):

        # Boston housing dataset
        boston = load_boston()
        #X = pd.DataFrame(boston.data, columns=boston.feature_names)
        #Y = pd.Series(boston.target, name='response')
        X = pd.DataFrame(boston.data, columns=boston.feature_names).loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
        Y = pd.Series(boston.target, name='response').loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
        df = pd.concat( [X,Y], axis=1 )

        st.markdown('The Boston housing dataset is used as the example.')
        st.write(df.head(5))

        build_model(df)


with st.sidebar.subheader('Created by:'):
    st.sidebar.markdown('''[Umang Bagadi](https://github.com/umangbagadi03)''')




    

