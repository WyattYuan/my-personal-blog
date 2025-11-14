import { getCollection, type CollectionEntry } from 'astro:content';

export interface Category {
  slug: string;
  displayName: string;
  description?: string;
  icon?: string;
}

// 分类显示名映射
export const CATEGORY_MAP: Record<string, Category> = {
  'web-dev': {
    slug: 'web-dev',
    displayName: 'Web开发',
    description: '前端、后端和全栈开发技术',
    icon: '🌐'
  },
  'hardware': {
    slug: 'hardware',
    displayName: '硬件开发',
    description: 'FPGA、Verilog等硬件描述语言',
    icon: '⚡'
  },
  'data-science': {
    slug: 'data-science',
    displayName: '数据科学',
    description: '数据分析、机器学习和AI',
    icon: '📊'
  },
  'tools': {
    slug: 'tools',
    displayName: '工具配置',
    description: '开发环境和工具使用技巧',
    icon: '🔧'
  },
};

/**
 * 从文章ID提取分类slug
 * 例如: "hardware/verilog-setup" -> "hardware"
 */
export function getCategoryFromId(id: string): string {
  const parts = id.split('/');
  return parts.length > 1 ? parts[0] : 'uncategorized';
}

/**
 * 获取分类的显示信息
 */
export function getCategoryInfo(slug: string): Category {
  return CATEGORY_MAP[slug] || {
    slug,
    displayName: slug,
    description: '',
    icon: '📁'
  };
}

/**
 * 获取所有存在的分类
 */
export async function getAllCategories(): Promise<Category[]> {
  const posts = await getCollection('blog');
  const categorySet = new Set(posts.map(p => getCategoryFromId(p.id)));

  return Array.from(categorySet)
    .map(slug => getCategoryInfo(slug))
    .sort((a, b) => a.displayName.localeCompare(b.displayName, 'zh-CN'));
}

/**
 * 按分类分组文章
 */
export async function getPostsByCategory(): Promise<Map<string, CollectionEntry<'blog'>[]>> {
  const posts = await getCollection('blog');
  const grouped = new Map<string, CollectionEntry<'blog'>[]>();

  // 按发布日期排序
  const sortedPosts = posts.sort(
    (a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf()
  );

  sortedPosts.forEach(post => {
    const category = getCategoryFromId(post.id);
    if (!grouped.has(category)) {
      grouped.set(category, []);
    }
    grouped.get(category)!.push(post);
  });

  return grouped;
}

/**
 * 获取指定分类的所有文章
 */
export async function getPostsInCategory(categorySlug: string): Promise<CollectionEntry<'blog'>[]> {
  const posts = await getCollection('blog');

  return posts
    .filter(post => getCategoryFromId(post.id) === categorySlug)
    .sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf());
}

/**
 * 获取分类的文章数量
 */
export async function getCategoryCounts(): Promise<Map<string, number>> {
  const posts = await getCollection('blog');
  const counts = new Map<string, number>();

  posts.forEach(post => {
    const category = getCategoryFromId(post.id);
    counts.set(category, (counts.get(category) || 0) + 1);
  });

  return counts;
}
