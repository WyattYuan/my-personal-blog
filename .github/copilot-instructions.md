# AI Coding Agent Instructions for Wyatt Yuan's Personal Blog

## Project Overview
This is an Astro-based personal blog featuring content collections, MDX support, and a clean Bear Blog-inspired design. The site serves as a portfolio and writing platform with RSS feed and sitemap generation.

## Architecture & Key Patterns

### Content Management
- **Content Collections**: Blog posts stored in `src/content/blog/` with schema validation in `src/content.config.ts`
- **Frontmatter Schema**: Required fields include `title`, `description`, `pubDate`; optional `updatedDate` and `heroImage`
- **Dynamic Routing**: Blog posts use `[...slug].astro` with `getStaticPaths()` for static generation
- **MDX Support**: Full Markdown + JSX support for rich content

### Component Architecture
- **Layout System**: `BlogPost.astro` layout handles post rendering with hero images, metadata, and responsive design
- **Reusable Components**: Header, Footer, FormattedDate, HeaderLink components in `src/components/`
- **Styling Approach**: Scoped CSS with CSS custom properties for theming (accent colors, grays, box shadows)
- **Image Optimization**: Astro's `<Image>` component with width/height props for performance

### Key Files & Structure
```
src/
├── content.config.ts    # Content collection schemas
├── layouts/BlogPost.astro # Main blog post layout
├── pages/
│   ├── index.astro      # Homepage (currently minimal)
│   ├── blog/index.astro # Blog listing with card layout
│   └── blog/[...slug].astro # Dynamic post routing
├── components/          # Reusable UI components
└── styles/global.css    # Global styles with custom properties
```

## Development Workflow

### Essential Commands
- `npm run dev` - Start development server at localhost:4321
- `npm run build` - Generate production build to `./dist/`
- `npm run preview` - Preview production build locally
- `npm run astro check` - Type checking and validation

### Content Creation
1. Add new blog post to `src/content/blog/` with `.md` or `.mdx` extension
2. Include required frontmatter: `title`, `description`, `pubDate`
3. Optional: `heroImage` (place in `src/assets/`), `updatedDate`
4. Posts automatically generate routes at `/blog/{slug}/`

### Styling Conventions
- **CSS Custom Properties**: Use variables like `var(--accent)`, `var(--gray)`, `var(--box-shadow)`
- **Typography**: Atkinson font family loaded from `/fonts/` directory
- **Responsive Design**: Mobile-first with breakpoints around 720px
- **Color Scheme**: Blue accent (`#2337ff`) with gray scale for text hierarchy

## Integration Points

### Astro Integrations
- **@astrojs/mdx**: Markdown + JSX support
- **@astrojs/rss**: Automatic RSS feed generation at `/rss.xml`
- **@astrojs/sitemap**: SEO-friendly sitemap generation
- **sharp**: Image optimization and processing

### External Dependencies
- **Astro Image Component**: For optimized hero images with `width={720} height={360}` pattern
- **Font Loading**: WOFF fonts served from `/public/fonts/` directory

## Common Patterns & Conventions

### Blog Post Structure
```astro
---
// Frontmatter in .md/.mdx files
title: "Post Title"
description: "Brief description"
pubDate: 2024-01-15
heroImage: ../../assets/image.jpg
---

# Content in Markdown/MDX
```

### Component Props Pattern
```astro
---
// Component props follow TypeScript CollectionEntry pattern
type Props = CollectionEntry<'blog'>['data'];
const { title, description, pubDate, heroImage } = Astro.props;
---
```

### Navigation Links
- Use `HeaderLink` component for internal navigation
- Social links in Header component (currently commented out)
- Footer contains site credits and additional links

### Image Usage
- Hero images: `<Image width={1020} height={510} src={heroImage} alt="" />`
- Blog listing thumbnails: `<Image width={720} height={360} src={heroImage} alt="" />`
- All images use Astro's optimized Image component

## Quality Assurance
- **Type Safety**: Full TypeScript support with content collection schemas
- **SEO**: Automatic OpenGraph data, canonical URLs, and meta tags via `BaseHead.astro`
- **Performance**: 100/100 Lighthouse scores with optimized images and minimal CSS
- **Accessibility**: Semantic HTML, screen reader support, and keyboard navigation

## Deployment Notes
- Static site generation - no server runtime required
- Built output goes to `./dist/` directory
- RSS feed available at `/rss.xml` for feed readers
- Sitemap automatically generated for search engines</content>
<parameter name="filePath">d:\100Code\my-personal-blog\.github\copilot-instructions.md