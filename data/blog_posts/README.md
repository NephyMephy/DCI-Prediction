# DCI Blog System - Individual Post Files

The blog system has been restructured to use individual JSON files for each blog post, stored in the `data/blog_posts/` directory.

## File Structure

```
data/
└── blog_posts/
    ├── index.json          # Master index of all posts
    ├── 20250801-01.json   # Date-based individual post files
    ├── 20250728-01.json   # Format: YYYYMMDD-ID.json
    ├── 20250725-01.json
    ├── 20250722-01.json
    ├── 20250720-01.json
    └── 20250718-01.json
```

## Naming Convention

Blog post files use the format: `YYYYMMDD-ID.json`

- **YYYYMMDD**: Publication date (e.g., 20250801 for August 1, 2025)
- **ID**: Two-digit unique identifier for posts on the same date (01, 02, 03, etc.)

Examples:
- `20250801-01.json` - First post published on August 1, 2025
- `20250801-02.json` - Second post published on August 1, 2025
- `20250728-01.json` - First post published on July 28, 2025

## Blog Post File Format

Each blog post file follows this JSON structure:

```json
{
    "id": 1,
    "slug": "post-slug-name",
    "title": "Post Title",
    "excerpt": "Brief description of the post content",
    "content": "Full article content with proper formatting...",
    "category": "analysis|history|season|corps",
    "author": "Author Name",
    "date": "YYYY-MM-DD",
    "readTime": "X min read",
    "featured": true|false,
    "image": null|"image-filename.jpg",
    "tags": ["tag1", "tag2", "tag3"],
    "published": true|false
}
```

## Index File Format

The `index.json` file contains a summary of all available posts:

```json
{
    "posts": [
        {
            "filename": "20250801-01.json",
            "slug": "post-slug",
            "title": "Post Title",
            "date": "YYYY-MM-DD",
            "category": "category",
            "featured": true|false,
            "published": true|false
        }
    ],
    "lastUpdated": "YYYY-MM-DD"
}
```

## Adding New Blog Posts

1. **Determine the filename**: Use format `YYYYMMDD-ID.json` based on publication date
2. **Create the post file**: Create a new JSON file in `data/blog_posts/` with the date-based name
3. **Update the index**: Add the new post entry to `index.json` with the filename
4. **Set published status**: Set `"published": true` when ready to display

Example for adding a post on August 5, 2025:
- Filename: `20250805-01.json`
- Index entry includes: `"filename": "20250805-01.json"`

## Categories

- **analysis**: Data-driven analysis and statistical insights
- **history**: Historical perspectives and retrospectives
- **season**: Current season coverage and previews
- **corps**: Corps-specific content and spotlights

## Features

- **Individual file loading**: Each post loads independently for better performance
- **Flexible content management**: Easy to add, edit, or remove posts
- **Draft system**: Use `"published": false` for draft posts
- **Featured posts**: Mark important posts with `"featured": true`
- **Tag system**: Add relevant tags for better organization
- **SEO-friendly slugs**: Clean URLs for individual post pages (future feature)

## JavaScript Integration

The blog system automatically:
- Loads the index file first to get available posts and their filenames
- Fetches individual post files using the date-based filename format
- Handles missing or corrupted files gracefully
- Falls back to sample data if files are unavailable
- Sorts posts by date (newest first)
- Filters posts based on published status
- Provides functions to load posts by filename or slug

### Available Functions

- `loadBlogPost(filename)` - Load post by date-based filename
- `loadBlogPostBySlug(slug)` - Load post by slug (searches index)
- `getBlogPostFilenames()` - Get all available filenames

## Benefits of Date-Based Naming

- **Chronological organization**: Files are naturally sorted by date
- **Collision avoidance**: Multiple posts per day use incremental IDs
- **Archive-friendly**: Easy to identify post publication dates
- **Batch operations**: Easy to process posts by date ranges
- **File management**: Clear organization for content management
