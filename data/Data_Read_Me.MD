# DCI Scoring Data Documentation

## Overview

This directory contains comprehensive Drum Corps International (DCI) scoring data spanning from 1964 to 2025. The dataset includes detailed competition results, event information, corps performance data, and historical championship records. Note: 2020 data is not included due to the cancellation of the DCI season.

## Current Directory Structure

```
data/
├── Data_Read_Me.MD          # This documentation file
├── end_of_season.json       # End of season tournament data
├── finalists.json           # DCI World Championship finalists data
├── recent_updates.json      # Recent data updates and changes (internal use)
├── shows/                   # Event and show information
│   └── master_shows_list.csv
└── years/                   # Individual year data files (1964-2025, excluding 2020)
    ├── 1964_dci_data.csv
    ├── 1965_dci_data.csv
    ├── ...
    ├── 2019_dci_data.csv
    ├── 2021_dci_data.csv    # Note: 2020 missing due to season cancellation
    ├── 2022_dci_data.csv
    ├── ...
    └── 2025_dci_data.csv
```

## File Descriptions

### Core Data Files

#### `years/` Directory
Contains individual CSV files for each competition year from 1964-2025 (excluding 2020).

**File Format:** `YYYY_dci_data.csv`

**Standard Columns:**
- `Date` - Competition date (YYYY-MM-DD format)
- `Year` - Competition year
- `Month` - Competition month (MM format)
- `Day` - Competition day (DD format)
- `Month_Name` - Full month name
- `Season` - Season classification (Spring, Summer, Fall)
- `City` - Host city
- `State` - Host state/province
- `Location` - Combined city, state
- `Corps_Count` - Number of participating corps
- `Corps` - Corps name
- `Class` - Competition class (Open Class, World Class, Division I, etc.)
- `Place` - Final placement
- `Score` - Total score
- `Music` - Music subscore (when available)
- `Visual` - Visual subscore (when available)
- `General_Effect` - General Effect subscore (when available)
- `Penalties` - Penalty points deducted

**Note:** Column availability varies by year due to changes in DCI scoring systems over time.

#### `shows/master_shows_list.csv`
Complete database of all DCI competitions and events from 1964-2025.

**Columns:**
- `Date` - Event date (YYYY-MM-DD)
- `Year` - Event year
- `Month` - Event month (MM)
- `Day` - Event day (DD)
- `Month_Name` - Month name
- `Season` - Season type (Spring, Summer, Fall)
- `City` - Host city
- `State` - Host state/province
- `Location` - Full location string
- `Corps_Count` - Number of participating corps

#### `finalists.json`
Historical data of DCI World Championship finalists including placements and scores.

**Structure:**
```json
{
  "dci_world_class_finalists": [
    {
      "year": 1972,
      "class": "Open",
      "finalists": [
        {"place": 1, "corps": "Anaheim Kingsmen", "score": 88.10},
        {"place": 2, "corps": "Blue Stars", "score": 87.45},
        {"place": 3, "corps": "Vanguard", "score": 87.35},
        {"place": 4, "corps": "27th Lancers", "score": 86.00},
......        
```

#### `end_of_season.json`
End of season tournament and championship data including semifinals and finals results.

#### `recent_updates.json`
Internal tracking file for recent changes and updates to the dataset. Not included in public data explorer.

## Data Coverage and Completeness

### Year Coverage
- **1964-2019**: Complete historical data
- **2020**: No data (DCI season cancelled due to COVID-19)
- **2021-2025**: Current and ongoing seasons

### Known Limitations
- Early years (1964-1971) may have incomplete scoring details
- Some regional competitions not included in comprehensive dataset
- Scoring system changes over time affect direct year-to-year comparisons
- Weather-related modifications and cancellations noted where possible
- Full recaps recorded but missing standardization (COMING SOON)
- Misattribution of circuit and class designations for corps (2025 correct, rolling corrections coming over next few weeks)

## Planned Future Additions

### Short Term (2025)
- **JSON Recaps Directory** (`json_recaps/`) - Structured JSON format of competition recaps
- **Corps History Files** - Individual corps performance histories
- **Venue Information** - Detailed venue and geographic data
- **Corps Logos** - Continued retreival and restoration of historic logos to at least 1000x1000 @ 300dpi

### Medium Term (2025-2026)
- **International Competitions** - Data from international drum corps events such as DCE, DCUK, and DCJ
- **Individual Caption Scores** - Detailed breakdown by judging caption
- **Weather Data Integration** - Weather conditions for outdoor competitions

### Long Term (2026+)
- **Multimedia Integration** - Links to performance videos and audio
- **Real-time Updates** - Live scoring integration during active season
- **Statistical Derivatives** - Pre-calculated statistics and trends
- **Corps Alumni Data** - Historical membership and staff information

## Technical Specifications

### File Formats
- **CSV Files**: UTF-8 encoded, comma-separated values with headers
- **JSON Files**: UTF-8 encoded, properly formatted JSON structure
- **Markdown**: UTF-8 encoded documentation files

### Data Consistency
- All dates use ISO 8601 format (YYYY-MM-DD)
- Scores are decimal values to three places (e.g., 95.650)
- Missing data represented as empty strings
- Corps names standardized across years

### File Sizes (Approximate)
- Individual year files: 10KB - 500KB each
- Master shows list: ~800KB
- JSON files: 5KB - 100KB each
- Total current dataset: ~35MB

## Data Sources and Methodology

### Primary Sources
- Official DCI competition results and scoresheets
- DCI and DCA historical archives and documentation
- Competition programs and published results
- Corps websites and historical records

### Data Processing
- Automated collection where possible
- Manual verification of historical data
- Cross-referencing between multiple sources
- Regular data validation and cleanup processes

### Quality Assurance
- Duplicate detection and removal
- Score validation against known ranges
- Date and location consistency checks
- Corps name standardization

## Usage Guidelines

### Academic and Research Use
This dataset is suitable for:
- Statistical analysis of competitive trends
- Historical research on drum corps activity
- Performance analytics and scoring patterns
- Geographic analysis of competition distribution
- Longitudinal studies of competitive evolution

### Data Processing Recommendations
- Account for scoring system changes when comparing across eras
- Use Corps_Count field to understand competition size context
- Consider seasonal variations in competition scheduling
- Validate critical data points against official sources

### Citation and Attribution
When using this dataset in research or publications:
- Cite as "DCI Scoring History Dataset, compiled [current year]"
- Acknowledge the original source of DCI competition data
- Note any limitations relevant to your specific analysis

## Maintenance and Updates

### Update Schedule
- New competition results added within 24-48 hours during active season
- Historical corrections incorporated as discovered
- Annual comprehensive data review and validation
- Documentation updated with structural changes

### Version Control
- Major changes documented in `recent_updates.json`
- File modification dates indicate last update
- Semantic versioning for significant structural changes

### Data Integrity
- Regular automated validation checks
- Cross-file consistency verification
- Backup and recovery procedures in place
- Change tracking for audit purposes

## Contact and Contributions

### Reporting Issues
- Data corrections: Provide specific event details and sources
- Missing data: Include competition name, date, and location
- Technical issues: Describe the problem with steps to reproduce

### Data Contributions
- Historical data welcomed with verification
- Source documentation required for additions
- Format guidelines available for contributors

## Legal and Disclaimers

### Data Usage Rights
- Compiled for educational and research purposes
- Original competition results remain property of DCI and DCA
- No commercial use without appropriate permissions
- Attribution requested for academic and research use

### Accuracy Statement
While comprehensive efforts ensure data accuracy, this dataset may contain errors. Users should verify critical information against official DCI records when precision is essential for their use case.

### Privacy Considerations
- Only publicly available competition results included
- No personal information beyond publicly performed results
- Individual performer names not included in datasets

---

**Dataset Information:**
- **Last Updated:** August 2025
- **Version:** 2025.8.4
- **Total Competitions:** ~8,700 events
- **Total Performance Records:** ~500,000+ individual scores
- **Years Covered:** 1964-2025 (excluding 2020)
- **Data Format:** CSV primary, JSON supplementary