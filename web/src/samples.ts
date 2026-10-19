import { Coffee, Rocket, Store, type LucideIcon } from "lucide-react";

export interface FieldDef {
  id: string;
  label: string;
  hint: string;
}

export const INCOME_FIELDS: FieldDef[] = [
  { id: "revenue", label: "Revenue this year", hint: "Total sales before any costs" },
  { id: "revenue_prior", label: "Revenue last year", hint: "Same figure for the previous year" },
  { id: "cogs", label: "Cost of goods sold", hint: "What you paid for the products or services you sold" },
  { id: "operating_expenses", label: "Operating expenses", hint: "Rent, salaries, marketing, software…" },
  { id: "interest_expense", label: "Interest paid", hint: "Interest on loans and credit lines" },
  { id: "net_income", label: "Net profit", hint: "The bottom line after every cost and tax" },
];

export const BALANCE_FIELDS: FieldDef[] = [
  { id: "cash", label: "Cash in the bank", hint: "Cash and cash equivalents" },
  { id: "inventory", label: "Inventory", hint: "Stock you still have to sell" },
  { id: "current_assets", label: "Current assets", hint: "Cash, inventory and money customers owe you" },
  { id: "current_liabilities", label: "Current liabilities", hint: "Bills and debt due within a year" },
  { id: "total_liabilities", label: "Total liabilities", hint: "Everything the business owes" },
  { id: "equity", label: "Owner's equity", hint: "What the business is worth to its owners on paper" },
];

export interface Sample {
  icon: LucideIcon;
  name: string;
  fields: Record<string, number>;
}

export const SAMPLES: Sample[] = [
  {
    icon: Coffee,
    name: "Coffee shop",
    fields: {
      revenue: 510000, revenue_prior: 450000, cogs: 204000, operating_expenses: 230000, interest_expense: 12000,
      net_income: 48000, cash: 60000, inventory: 35000, current_assets: 140000, current_liabilities: 90000,
      total_liabilities: 210000, equity: 175000,
    },
  },
  {
    icon: Rocket,
    name: "SaaS startup",
    fields: {
      revenue: 2400000, revenue_prior: 1500000, cogs: 480000, operating_expenses: 2300000, interest_expense: 0,
      net_income: -420000, cash: 1800000, current_assets: 2300000, current_liabilities: 900000,
      total_liabilities: 1100000, equity: 1600000,
    },
  },
  {
    icon: Store,
    name: "Struggling retailer",
    fields: {
      revenue: 880000, revenue_prior: 960000, cogs: 690000, operating_expenses: 215000, interest_expense: 28000,
      net_income: -51000, cash: 18000, inventory: 160000, current_assets: 230000, current_liabilities: 260000,
      total_liabilities: 610000, equity: 140000,
    },
  },
];

export const PASTE_EXAMPLE = `Revenue            510,000   450,000
Cost of goods sold (204,000)
Operating expenses 230,000
Net income          48,000
Total current assets      140,000
Total current liabilities  90,000`;
