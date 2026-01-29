"""
Keyword Filter Streamlit App
Filters Search Console Queries by keywords and displays results
"""

import streamlit as st
import pandas as pd
import re
from io import BytesIO
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# Page configuration
st.set_page_config(page_title="Keyword Filter Tool", page_icon="🔍", layout="wide")


def filter_queries_by_keywords(df, keywords):
    """
    Filters queries by keywords and creates two DataFrames
    Keywords must match as whole words (word boundaries are considered)

    Args:
        df: DataFrame with Search Console data
        keywords (list): List of keywords to filter by

    Returns:
        tuple: (df1, df2)
            - df1: DataFrame with columns [keyword_cluster, query, clicks, impressions]
            - df2: DataFrame with columns [keyword_cluster, sum_clicks, sum_impressions, query_count]
    """
    # List for results
    results = []

    # Find all matching queries for each keyword
    for keyword in keywords:
        keyword_lower = keyword.lower()

        # Create regex pattern for word boundaries
        pattern = r"\b" + re.escape(keyword_lower) + r"\b"

        # Filter queries that contain this keyword as a whole word
        mask = df["Query"].str.lower().str.contains(pattern, na=False, regex=True)
        matching_queries = df[mask]

        # Create an entry with the keyword cluster for each matching query
        for _, row in matching_queries.iterrows():
            results.append(
                {
                    "keyword_cluster": keyword,
                    "query": row["Query"],
                    "clicks": row["Clicks"],
                    "impressions": row["Impressions"],
                }
            )

    # df1: Create detail DataFrame
    df1 = pd.DataFrame(results)

    # df2: Aggregated sums per keyword_cluster
    if len(df1) > 0:
        df2 = (
            df1.groupby("keyword_cluster")
            .agg({"clicks": "sum", "impressions": "sum", "query": "count"})
            .reset_index()
        )

        # Rename columns
        df2.columns = [
            "keyword_cluster",
            "sum_clicks",
            "sum_impressions",
            "query_count",
        ]

        # Sort by impressions
        df2 = df2.sort_values("sum_impressions", ascending=False)
    else:
        df2 = pd.DataFrame(
            columns=["keyword_cluster", "sum_clicks", "sum_impressions", "query_count"]
        )

    return df1, df2


@st.cache_data
def convert_df_to_csv(df):
    """Converts DataFrame to CSV for download"""
    return df.to_csv(index=False).encode("utf-8")


@st.cache_data
def convert_df_to_excel(df1, df2):
    """Converts both DataFrames to Excel with two sheets"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df1.to_excel(writer, sheet_name="Details", index=False)
        df2.to_excel(writer, sheet_name="Summary", index=False)
    return output.getvalue()


def create_word_cloud(df1, weight_by="impressions", top_n=100, stopwords=None):
    """
    Creates Word Cloud from queries

    Args:
        df1: DataFrame with query details
        weight_by: 'impressions' or 'clicks' - what to weight words by
        top_n: Number of top words to display
        stopwords: List of words to exclude

    Returns:
        matplotlib figure
    """
    # Default stopwords (English)
    default_stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "been",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "must",
        "can",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "now",
        "also",
        "here",
        "there",
        "then",
    }

    if stopwords:
        default_stopwords.update(set(stopwords))

    # Collect words with weighting
    word_weights = Counter()

    for _, row in df1.iterrows():
        query = row["query"]
        weight = row[weight_by]

        # Split query into individual words
        words = query.lower().split()

        for word in words:
            # Remove special characters
            word = re.sub(r"[^\w\s]", "", word)

            # Filter stopwords and short words
            if word and len(word) > 2 and word not in default_stopwords:
                word_weights[word] += weight

    # Take top N words
    top_words = dict(word_weights.most_common(top_n))

    if not top_words:
        return None

    # Create Word Cloud
    wordcloud = WordCloud(
        width=1600,
        height=800,
        background_color="white",
        colormap="viridis",
        relative_scaling=0.5,
        min_font_size=10,
    ).generate_from_frequencies(top_words)

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)

    return fig


def main():
    # Header
    st.title("🔍 Keyword Filter Tool")
    st.markdown("Filter Search Console queries by keywords and analyze the results")

    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")

    # File Upload
    st.sidebar.subheader("1. Upload CSV File")
    uploaded_file = st.sidebar.file_uploader(
        "Choose your Search Console CSV file",
        type=["csv"],
        help="The CSV should contain the columns Query, Clicks, Impressions, CTR, Avg Position",
    )

    # Keywords Input
    st.sidebar.subheader("2. Enter Keywords")
    keywords_input = st.sidebar.text_area(
        "Keywords (one keyword per line)",
        height=150,
        placeholder="training\neducation\ncourse\nprogram",
        help="Each keyword is searched as a whole word (e.g. 'training' won't match 'trainings')",
    )

    # Filter Button
    filter_button = st.sidebar.button(
        "🔍 Filter", type="primary", use_container_width=True
    )

    # Main Content Area
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)

            # Validation
            required_columns = ["Query", "Clicks", "Impressions"]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                st.error(f"❌ Missing columns: {', '.join(missing_columns)}")
                st.info(
                    "The CSV must contain at least the following columns: Query, Clicks, Impressions"
                )
                return

            # Display file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Queries", f"{len(df):,}")
            with col2:
                st.metric("👆 Total Clicks", f"{df['Clicks'].sum():,}")
            with col3:
                st.metric("👁️ Total Impressions", f"{df['Impressions'].sum():,}")

            st.markdown("---")

            # If Filter Button was clicked
            if filter_button or "df1" in st.session_state:
                # Process keywords
                keywords = [
                    kw.strip()
                    for kw in keywords_input.strip().split("\n")
                    if kw.strip()
                ]

                if not keywords:
                    st.warning("⚠️ Please enter at least one keyword!")
                    return

                # Filter
                with st.spinner("Filtering queries..."):
                    df1, df2 = filter_queries_by_keywords(df, keywords)

                    # Store in session state
                    st.session_state["df1"] = df1
                    st.session_state["df2"] = df2
                    st.session_state["keywords"] = keywords

            # If results are available
            if "df1" in st.session_state and "df2" in st.session_state:
                df1 = st.session_state["df1"]
                df2 = st.session_state["df2"]
                keywords = st.session_state["keywords"]

                if len(df1) == 0:
                    st.warning("⚠️ No queries found with the specified keywords!")
                    return

                # Results header
                st.success(
                    f"✅ {len(df1):,} queries found for {len(keywords)} keywords"
                )

                # Tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(
                    ["📊 Summary", "📋 Details", "📈 Statistics", "☁️ Word Cloud"]
                )

                # Tab 1: Summary (df2)
                with tab1:
                    st.subheader("Aggregated Results per Keyword Cluster")

                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🔑 Keywords", len(df2))
                    with col2:
                        st.metric("👆 Total Clicks", f"{df2['sum_clicks'].sum():,}")
                    with col3:
                        st.metric(
                            "👁️ Total Impressions", f"{df2['sum_impressions'].sum():,}"
                        )

                    st.markdown("---")

                    # Display DataFrame
                    st.dataframe(
                        df2,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "keyword_cluster": st.column_config.TextColumn(
                                "Keyword Cluster", width="medium"
                            ),
                            "sum_clicks": st.column_config.NumberColumn(
                                "Clicks", format="%d"
                            ),
                            "sum_impressions": st.column_config.NumberColumn(
                                "Impressions", format="%d"
                            ),
                            "query_count": st.column_config.NumberColumn(
                                "Number of Queries", format="%d"
                            ),
                        },
                    )

                    # Bar Chart
                    st.subheader("Impressions per Keyword")
                    st.bar_chart(df2.set_index("keyword_cluster")["sum_impressions"])

                # Tab 2: Details (df1)
                with tab2:
                    st.subheader("Detailed Query List")

                    # Filter for keyword cluster
                    selected_cluster = st.selectbox(
                        "Filter by Keyword Cluster:",
                        options=["All"]
                        + sorted(df1["keyword_cluster"].unique().tolist()),
                    )

                    # Filter DataFrame if necessary
                    if selected_cluster != "All":
                        df1_filtered = df1[df1["keyword_cluster"] == selected_cluster]
                    else:
                        df1_filtered = df1

                    st.info(f"Showing {len(df1_filtered):,} of {len(df1):,} queries")

                    # Display DataFrame
                    st.dataframe(
                        df1_filtered,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "keyword_cluster": st.column_config.TextColumn(
                                "Keyword Cluster", width="small"
                            ),
                            "query": st.column_config.TextColumn(
                                "Query", width="large"
                            ),
                            "clicks": st.column_config.NumberColumn(
                                "Clicks", format="%d"
                            ),
                            "impressions": st.column_config.NumberColumn(
                                "Impressions", format="%d"
                            ),
                        },
                    )

                # Tab 3: Statistics
                with tab3:
                    st.subheader("Detailed Statistics")

                    # Statistics per keyword
                    for keyword in keywords:
                        with st.expander(f"📊 {keyword}"):
                            kw_data = df1[df1["keyword_cluster"] == keyword]

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Queries", f"{len(kw_data):,}")
                            with col2:
                                st.metric("Clicks", f"{kw_data['clicks'].sum():,}")
                            with col3:
                                st.metric(
                                    "Impressions", f"{kw_data['impressions'].sum():,}"
                                )
                            with col4:
                                avg_ctr = (
                                    (
                                        kw_data["clicks"].sum()
                                        / kw_data["impressions"].sum()
                                        * 100
                                    )
                                    if kw_data["impressions"].sum() > 0
                                    else 0
                                )
                                st.metric("Avg CTR", f"{avg_ctr:.2f}%")

                            # Top 5 Queries
                            st.markdown("**Top 5 Queries:**")
                            top_queries = kw_data.nlargest(5, "impressions")[
                                ["query", "impressions", "clicks"]
                            ]
                            st.dataframe(
                                top_queries, hide_index=True, use_container_width=True
                            )

                # Tab 4: Word Cloud
                with tab4:
                    st.subheader("☁️ Word Cloud")
                    st.markdown(
                        "Visualization of the most frequent words in the filtered queries"
                    )

                    # Settings for Word Cloud
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        weight_by = st.selectbox(
                            "Weight by:",
                            options=["impressions", "clicks"],
                            help="Choose whether words are weighted by impressions or clicks",
                        )

                    with col2:
                        top_n = st.slider(
                            "Number of Words:",
                            min_value=20,
                            max_value=200,
                            value=100,
                            step=10,
                            help="How many top words should be displayed?",
                        )

                    with col3:
                        st.markdown("&nbsp;")  # Spacer
                        filter_keywords_option = st.checkbox(
                            "Hide keywords",
                            value=True,
                            help="Filters out the entered keywords from the word cloud",
                        )

                    # Additional stopwords
                    with st.expander("⚙️ Advanced Settings"):
                        custom_stopwords = st.text_area(
                            "Additional words to exclude (one word per line):",
                            height=100,
                            placeholder="2025\n2026\ncom",
                            help="These words will be additionally removed from the word cloud",
                        )

                    # Compile stopwords
                    stopwords_list = []
                    if filter_keywords_option:
                        stopwords_list.extend([kw.lower() for kw in keywords])
                    if custom_stopwords:
                        stopwords_list.extend(
                            [
                                w.strip().lower()
                                for w in custom_stopwords.split("\n")
                                if w.strip()
                            ]
                        )

                    # Create Word Cloud
                    with st.spinner("Creating Word Cloud..."):
                        fig = create_word_cloud(
                            df1,
                            weight_by=weight_by,
                            top_n=top_n,
                            stopwords=stopwords_list,
                        )

                        if fig:
                            st.pyplot(fig)

                            # Info box
                            st.info(
                                f"""
                            💡 **Legend:**
                            - The larger the word, the higher the {weight_by.title()}
                            - Top {top_n} words are displayed
                            - Stopwords and short words (<3 characters) are hidden
                            """
                            )

                            # Top 10 words as table
                            st.markdown("---")
                            st.subheader("📊 Top 10 Words")

                            # Recalculate words for table
                            word_weights = Counter()
                            for _, row in df1.iterrows():
                                query = row["query"]
                                weight = row[weight_by]
                                words = query.lower().split()
                                for word in words:
                                    word = re.sub(r"[^\w\s]", "", word)
                                    if (
                                        word
                                        and len(word) > 2
                                        and word not in stopwords_list
                                    ):
                                        word_weights[word] += weight

                            top_words_df = pd.DataFrame(
                                word_weights.most_common(10),
                                columns=["Word", f"Total {weight_by.title()}"],
                            )
                            st.dataframe(
                                top_words_df, hide_index=True, use_container_width=True
                            )
                        else:
                            st.warning(
                                "⚠️ No words found for Word Cloud. Try adjusting the stopword filters."
                            )

                # Download section
                st.markdown("---")
                st.subheader("💾 Download Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    # Download df1 as CSV
                    csv1 = convert_df_to_csv(df1)
                    st.download_button(
                        label="📥 Details CSV",
                        data=csv1,
                        file_name="keyword_filter_details.csv",
                        mime="text/csv",
                        help="Downloads the detailed query list",
                    )

                with col2:
                    # Download df2 as CSV
                    csv2 = convert_df_to_csv(df2)
                    st.download_button(
                        label="📥 Summary CSV",
                        data=csv2,
                        file_name="keyword_filter_summary.csv",
                        mime="text/csv",
                        help="Downloads the aggregated summary",
                    )

                with col3:
                    # Download both as Excel
                    excel_data = convert_df_to_excel(df1, df2)
                    st.download_button(
                        label="📥 Excel (both sheets)",
                        data=excel_data,
                        file_name="keyword_filter_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Downloads both DataFrames as Excel file with two sheets",
                    )

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)

    else:
        # Instructions when no file is uploaded
        st.info("👈 Please upload a CSV file in the sidebar to get started")

        st.markdown("### 📝 Instructions")
        st.markdown(
            """
        1. **Upload CSV file**: Upload your Search Console CSV file (with columns: Query, Clicks, Impressions)
        2. **Enter keywords**: Enter your keywords (one keyword per line)
        3. **Filter**: Click the "Filter" button
        4. **View results**: Navigate through the tabs to see different views
        5. **Word Cloud**: Visualize the most frequent words weighted by impressions or clicks
        6. **Download**: Download the filtered results as CSV or Excel
        
        #### 💡 Notes:
        - Keywords are searched as **whole words** (e.g. "training" won't match "trainings")
        - The search is **case-insensitive** (uppercase/lowercase doesn't matter)
        - Each query can match multiple keywords
        - The **Word Cloud** shows the most important words based on their weighting
        """
        )


if __name__ == "__main__":
    main()
