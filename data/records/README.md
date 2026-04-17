# Records Data Structure

This directory contains JSON files with records data for different drum corps circuits.

## File Structure

### `records_index.json`
Master index file listing all available circuit record files.

```json
{
  "circuits": [
    {
      "code": "DCI",
      "name": "Drum Corps International", 
      "file": "dci_records.json",
      "active": true
    }
  ],
  "lastUpdated": "2025-08-11",
  "version": "1.0"
}
```

### Individual Circuit Files
Each circuit has its own JSON file with the following structure:

```json
{
  "circuit": "DCI",
  "circuitName": "Drum Corps International",
  "records": [
    {
      "category": "Record category name",
      "holder": "Corps name",
      "detail": "Record details/value",
      "year": "Year(s) achieved",
      "location": "Location where achieved"
    }
  ],
  "topScores": [
    {
      "rank": 1,
      "score": "99.65",
      "corps": "Blue Devils",
      "year": "2014",
      "location": "Indianapolis, IN"
    }
  ]
}
```

## Available Circuit Files

- `dci_records.json` - Drum Corps International
- `dca_records.json` - Drum Corps Associates  
- `vfw_records.json` - Veterans of Foreign Wars
- `cyo_records.json` - Catholic Youth Organization
- `al_records.json` - American Legion

## Top Scores Data

The `topScores` array contains the highest scores ever achieved in each circuit, ranked from highest to lowest. For circuits with ties, multiple entries may share the same rank.

### DCI Top 10 World Class Scores
1. 99.65 - Blue Devils (2014)
2. 99.15 - Cadets (2005) & Cavaliers (2002)
4. 99.05 - Blue Devils (2009)
5. 98.975 - Blue Devils (2023)
6. 98.9 - Blue Devils (2010)
7. 98.8 - Blue Devils (2003) & Santa Clara Vanguard (1989)
9. 98.75 - Blue Devils (2022) & Bluecoats (2024)
11. 98.7 - Blue Devils (2012) & Cavaliers (2004)
13. 98.625 - Santa Clara Vanguard (2018)
14. 98.538 - Blue Devils (2017)

## Usage in JavaScript

The records page JavaScript loads this data dynamically:

```javascript
// Load records data
const response = await fetch('../data/records/dci_records.json');
const data = await response.json();

// Access records
const records = data.records;
const topScores = data.topScores;
```

## Data Maintenance

Use the `generate_records_data.py` script to:
- Analyze CSV data files
- Generate top scores lists
- Update records files automatically

```bash
python generate_records_data.py
```

## Record Categories

### Common Record Types
- **Highest Score Ever** - Single highest score achieved
- **Championship Streaks** - Consecutive titles won  
- **Medal Streaks** - Consecutive top-3 finishes
- **Finalist Streaks** - Consecutive finals appearances
- **Win Streaks** - Consecutive competition wins
- **Caption Records** - Individual caption achievements

### Data Sources
Records data is compiled from:
- Official DCI/DCA results
- Historical competition data
- Verified performance records
- Championship finals results

Last Updated: August 11, 2025
