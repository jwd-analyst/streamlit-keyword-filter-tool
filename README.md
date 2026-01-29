# 🔍 Keyword Filter Streamlit App

An interactive web app for filtering and analyzing Google Search Console queries by keywords.

## Features

✨ **Interactive User Interface**
- CSV file upload via drag & drop
- Easy keyword entry (one keyword per line)
- Live filtering with one click

📊 **Four Interactive Tabs**
- **Summary**: Aggregated data per keyword with visualization
- **Details**: Complete query list with filter options
- **Statistics**: Detailed metrics per keyword including top queries
- **Word Cloud**: Visualization of most frequent words weighted by impressions or clicks

💾 **Flexible Downloads**
- Details as CSV
- Summary as CSV
- Both DataFrames as Excel with two sheets

🎯 **Intelligent Matching**
- Word boundary-based search (whole word)
- Case-insensitive (uppercase/lowercase doesn't matter)
- "training" does NOT match "trainings"

☁️ **Word Cloud Features**
- Weighting by impressions or clicks
- Adjustable number of displayed words (20-200)
- Automatic stopword filtering (English words)
- Custom word exclusions
- Option to hide filter keywords
- Top 10 words as table

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python Package Manager)

### Install Packages

```bash
pip install streamlit pandas openpyxl wordcloud matplotlib
```

Or with requirements.txt:

```bash
pip install -r requirements.txt
```

## Starting the App

### Method 1: Via Terminal/Command Line

```bash
streamlit run streamlit_app_en.py
```

### Method 2: With Python

```bash
python -m streamlit run streamlit_app_en.py
```

The app will automatically open in your browser at `http://localhost:8501`

## Usage

### Step 1: Upload CSV File
- Click "Browse files" in the sidebar
- Select your Search Console CSV file
- **Required columns**: Query, Clicks, Impressions

### Step 2: Enter Keywords
- Enter your keywords in the text field
- One keyword per line
- Example:
  ```
  training
  education
  course
  program
  ```

### Step 3: Filter
- Click the "🔍 Filter" button
- The app processes your data

### Step 4: View Results
Navigate through the four tabs:

**📊 Summary**
- Aggregated metrics per keyword
- Overview: Clicks, Impressions, Query Count
- Bar chart of impressions

**📋 Details**
- All filtered queries
- Filter by keyword cluster
- Complete data table

**📈 Statistics**
- Detailed metrics per keyword
- Top 5 queries per keyword
- CTR calculations

**☁️ Word Cloud**
- Visualization of most frequent words
- Words are weighted by impressions or clicks
- The larger the word, the higher the weighting
- Adjustable options:
  - Weighting: Impressions or Clicks
  - Number of words: 20-200
  - Hide keywords
  - Custom stopwords
- Top 10 words as data table

### Step 5: Use Word Cloud

Switch to the "☁️ Word Cloud" tab:

**Adjust Settings:**
- **Weighting**: Choose whether words are weighted by impressions or clicks
- **Number of Words**: Slider from 20-200 (default: 100)
- **Hide Keywords**: Filters out your entered keywords from the cloud
- **Additional Stopwords**: Add your own words to be hidden (e.g., year numbers)

**Interpretation:**
- Larger words = higher impressions/clicks
- The word cloud shows at a glance which themes are dominant in your filtered queries
- The top 10 table below shows the exact values

**Example Stopwords:**
```
2025
2026
com
www
```

### Step 6: Download
Three download options available:
- **Details CSV**: All queries with keyword cluster
- **Summary CSV**: Aggregated data per keyword
- **Excel**: Both DataFrames in one file (two sheets)

## CSV Format

Your CSV file should contain at least these columns:

| Query | Clicks | Impressions | CTR | Avg Position |
|-------|--------|-------------|-----|--------------|
| training berlin | 8925 | 89285 | 10% | 1.9 |
| training hamburg | 5013 | 51525 | 9.73% | 2.05 |

**Required fields:**
- `Query` - Search term
- `Clicks` - Number of clicks
- `Impressions` - Number of impressions

Optional fields are ignored.

## Output Structure

### DataFrame 1 (Details)
```
keyword_cluster | query | clicks | impressions
```

### DataFrame 2 (Summary)
```
keyword_cluster | sum_clicks | sum_impressions | query_count
```

## Example

**Input:**
- CSV with 50,000 queries
- Keywords: `berlin`, `hamburg`, `munich`

**Output:**
- df1: 5,336 filtered queries
- df2: 3 rows (one per keyword)
- Total: 1.79 million impressions

## Technical Details

### Keyword Matching
The app uses regex with word boundaries (`\b`):
```python
pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
```

**Examples:**
- ✅ `training` matches `training berlin`
- ✅ `training` matches `online training`
- ❌ `training` does NOT match `trainings`
- ❌ `training` does NOT match `retraining`

### Performance
- **Caching**: DataFrames are cached for faster downloads
- **Session State**: Results persist when navigating
- **Lazy Loading**: Downloads are only generated when needed

## Tips & Tricks

💡 **Large Files**: For files >100MB processing may take some time. The app shows a spinner during processing.

💡 **Test Keywords**: Start with a few keywords to see the results, then expand the list.

💡 **Browser Compatibility**: The app works best in Chrome, Firefox, or Edge.

💡 **Mobile Usage**: The app is responsive and works on tablets too.

## Troubleshooting

**"No module named 'streamlit'"**
```bash
pip install streamlit
```

**"No module named 'openpyxl'"**
```bash
pip install openpyxl
```

**"No module named 'wordcloud'"**
```bash
pip install wordcloud matplotlib
```

**App won't start**
- Check if port 8501 is available
- Start with `streamlit run streamlit_app_en.py --server.port 8502`

**CSV not recognized**
- Make sure the file is UTF-8 encoded
- Check if required columns are present

**No results**
- Check keyword spelling
- Test with known queries from your CSV

## Further Links

- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Streamlit Community Forum](https://discuss.streamlit.io)

## Support

For questions or problems:
1. Check troubleshooting above
2. Look in Streamlit documentation
3. Post in Streamlit Community Forum

## License

This project is open source and freely available.

## Author

Created with Claude (Anthropic) - 2025

---

**Good luck with your keyword analysis! 🚀**
