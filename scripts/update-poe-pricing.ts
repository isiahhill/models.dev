#!/usr/bin/env bun

import { readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";

const POE_API_URL = "https://api.poe.com/v1/models";
const BASE_DIR = path.join(process.cwd(), "providers", "poe", "models");

interface Pricing {
  prompt: string | null;
  completion: string | null;
  input_cache_read: string | null;
  input_cache_write: string | null;
  image?: string | null;
}

interface PoeModel {
  id: string;
  owned_by: string;
  pricing: Pricing | null;
}

type TomlData = Record<string, unknown>;

function normalizeCostValue(raw: string | null | undefined): number | null {
  if (!raw) return null;
  const num = Number(raw);
  if (Number.isNaN(num)) return null;
  const scaled = num * 1_000_000;
  // retain up to 6 decimal places to avoid truncating small values
  return Math.round(scaled * 1_000_000) / 1_000_000;
}

function normalizeTomlValue(value: string): string {
  const trimmed = value.trim();
  if (
    (trimmed.startsWith("\"") && trimmed.endsWith("\"")) ||
    (trimmed.startsWith("'") && trimmed.endsWith("'"))
  ) {
    return trimmed.slice(1, -1);
  }
  return trimmed;
}

function parseToml(content: string): TomlData {
  const data: TomlData = {};
  let section: string | null = null;
  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    if (line.startsWith("[")) {
      section = line.slice(1, -1);
      if (!data[section]) data[section] = {};
      continue;
    }
    const [key, ...rest] = line.split("=");
    if (!key || rest.length === 0) continue;
    const value = rest.join("=").trim();
    const normalized = normalizeTomlValue(value);
    const targetKey = key.trim();
    if (section) {
      const sectionData = data[section] as Record<string, unknown>;
      sectionData[targetKey] = normalized;
    } else {
      data[targetKey] = normalized;
    }
  }
  return data;
}

function formatCostValue(value: number | null): string | null {
  if (value === null) return null;
  const normalized = Math.round(value * 1_000_000) / 1_000_000;
  const formatted = normalized.toFixed(6).replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
  return formatted;
}

function updateCostSection(content: string, pricing: Pricing | null): string {
  const newLines: string[] = [];
  const lines = content.split(/\r?\n/);
  let inCostSection = false;
  for (const line of lines) {
    if (line.trim().startsWith("[cost]")) {
      inCostSection = true;
      newLines.push("[cost]");
      const costLines: string[] = [];
      const input = formatCostValue(normalizeCostValue(pricing?.prompt ?? null));
      const output = formatCostValue(normalizeCostValue(pricing?.completion ?? null));
      const cacheRead = formatCostValue(normalizeCostValue(pricing?.input_cache_read ?? null));
      const cacheWrite = formatCostValue(normalizeCostValue(pricing?.input_cache_write ?? null));
      const image = formatCostValue(normalizeCostValue(pricing?.image ?? null));

      if (input !== null) costLines.push(`input = ${input}`);
      if (output !== null) costLines.push(`output = ${output}`);
      if (cacheRead !== null) costLines.push(`cache_read = ${cacheRead}`);
      if (cacheWrite !== null) costLines.push(`cache_write = ${cacheWrite}`);
      if (image !== null) costLines.push(`image = ${image}`);
      newLines.push(...costLines);
      continue;
    }
    if (inCostSection) {
      if (line.startsWith("[")) {
        inCostSection = false;
        if (newLines.length > 0 && newLines[newLines.length - 1] !== "") {
          newLines.push("");
        }
        newLines.push(line);
      }
      continue;
    }
    newLines.push(line);
  }
  if (!content.includes("[cost]")) {
    const costLines: string[] = [];
    const input = formatCostValue(normalizeCostValue(pricing?.prompt ?? null));
    const output = formatCostValue(normalizeCostValue(pricing?.completion ?? null));
    const cacheRead = formatCostValue(normalizeCostValue(pricing?.input_cache_read ?? null));
    const cacheWrite = formatCostValue(normalizeCostValue(pricing?.input_cache_write ?? null));
    const image = formatCostValue(normalizeCostValue(pricing?.image ?? null));

    if (input !== null || output !== null || cacheRead !== null || cacheWrite !== null || image !== null) {
      const insertIndex = newLines.findIndex((line) => line.trim().startsWith("[limit]"));
      const block = ["[cost]"];
      if (input !== null) block.push(`input = ${input}`);
      if (output !== null) block.push(`output = ${output}`);
      if (cacheRead !== null) block.push(`cache_read = ${cacheRead}`);
      if (cacheWrite !== null) block.push(`cache_write = ${cacheWrite}`);
      if (image !== null) block.push(`image = ${image}`);
      if (insertIndex === -1) {
        if (newLines.length > 0 && newLines[newLines.length - 1] !== "") {
          newLines.push("");
        }
        newLines.push(...block, "");
      } else {
        newLines.splice(insertIndex, 0, ...block, "");
      }
    }
  }
  return newLines.join("\n");
}

async function fetchPoeModels(): Promise<PoeModel[]> {
  const headers: Record<string, string> = {};
  const apiKey = process.env.POE_API_KEY;
  if (apiKey) {
    headers.Authorization = `Bearer ${apiKey}`;
  }

  const response = await fetch(POE_API_URL, {
    headers,
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch Poe models: ${response.status} ${response.statusText}`);
  }
  const data = await response.json();
  return data.data as PoeModel[];
}

async function updatePricing() {
  const models = await fetchPoeModels();
  console.log(`Fetched ${models.length} models from Poe`);
  const dirEntries = await readdir(BASE_DIR, { withFileTypes: true });

  for (const entry of dirEntries) {
    if (!entry.isDirectory()) continue;
    const providerDir = path.join(BASE_DIR, entry.name);
    const tomlFiles = await readdir(providerDir, { withFileTypes: true });
    for (const fileEntry of tomlFiles) {
      if (!fileEntry.isFile() || !fileEntry.name.endsWith(".toml")) continue;
      const filePath = path.join(providerDir, fileEntry.name);
      const fileContent = await readFile(filePath, "utf8");
      const parsed = parseToml(fileContent);
      const name = parsed.name as string | undefined;
      if (!name) continue;

      const modelId = name.replace(/\s+/g, "-");
      const model = models.find((m) => m.id.toLowerCase() === modelId.toLowerCase());
      if (!model) continue;

      const updatedContent = updateCostSection(fileContent, model.pricing);
      if (updatedContent !== fileContent) {
        await writeFile(filePath, updatedContent, "utf8");
        console.log(`Updated pricing for ${filePath}`);
      }
    }
  }
}

await updatePricing();
