/**
 * Which series an economic release belongs to, matching eco_impact on the
 * server: "美国CPI月率(%)(月度)(十月)" → "美国CPI月率(%)(月度)".
 */

const PERIOD = /\s*\((?:[一二三四五六七八九十]+月|第[一二三四]季度|上半年|下半年|[0-9]+月[0-9]*日?当周|[0-9]+\/[0-9]+)\)$/;

export function seriesOf(event: string): string {
  return event.replace(PERIOD, "").trim();
}

/** Only US and China releases have a stored history. The Fed is not named 美国. */
const PREFIXES: Record<string, string[]> = {
  美国: ["美国", "美联储", "个人消费支出", "亚特兰大联储", "达拉斯联储", "ADP"],
  中国: ["中国"],
};

export function hasHistory(country: string, event: string): boolean {
  return (PREFIXES[country] ?? []).some((p) => event.startsWith(p));
}
