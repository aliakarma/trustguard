/* ═══════════════════════════════════════════════════════════════════════════
   TrustGuard Dashboard — Constants & Permission Definitions
   ═══════════════════════════════════════════════════════════════════════════ */

/* ── Android Dangerous Permissions (API 34) ───────────────────────────────── */
export const ANDROID_PERMISSIONS = [
  'READ_CALENDAR', 'WRITE_CALENDAR',
  'READ_CALL_LOG', 'WRITE_CALL_LOG', 'PROCESS_OUTGOING_CALLS',
  'CAMERA',
  'READ_CONTACTS', 'WRITE_CONTACTS', 'GET_ACCOUNTS',
  'ACCESS_FINE_LOCATION', 'ACCESS_COARSE_LOCATION', 'ACCESS_BACKGROUND_LOCATION',
  'RECORD_AUDIO',
  'READ_PHONE_STATE', 'READ_PHONE_NUMBERS', 'CALL_PHONE',
  'ANSWER_PHONE_CALLS', 'ADD_VOICEMAIL', 'USE_SIP',
  'BODY_SENSORS', 'BODY_SENSORS_BACKGROUND',
  'SEND_SMS', 'RECEIVE_SMS', 'READ_SMS', 'RECEIVE_WAP_PUSH', 'RECEIVE_MMS',
  'READ_EXTERNAL_STORAGE', 'WRITE_EXTERNAL_STORAGE',
  'READ_MEDIA_IMAGES', 'READ_MEDIA_VIDEO', 'READ_MEDIA_AUDIO',
  'BLUETOOTH_SCAN', 'BLUETOOTH_CONNECT', 'BLUETOOTH_ADVERTISE',
  'ACTIVITY_RECOGNITION',
  'NEARBY_WIFI_DEVICES',
  'POST_NOTIFICATIONS',
] as const;

export const NUM_PERMISSIONS = ANDROID_PERMISSIONS.length; // 37

export const PERMISSION_GROUPS: Record<string, string[]> = {
  Calendar: ['READ_CALENDAR', 'WRITE_CALENDAR'],
  'Call Log': ['READ_CALL_LOG', 'WRITE_CALL_LOG', 'PROCESS_OUTGOING_CALLS'],
  Camera: ['CAMERA'],
  Contacts: ['READ_CONTACTS', 'WRITE_CONTACTS', 'GET_ACCOUNTS'],
  Location: ['ACCESS_FINE_LOCATION', 'ACCESS_COARSE_LOCATION', 'ACCESS_BACKGROUND_LOCATION'],
  Microphone: ['RECORD_AUDIO'],
  Phone: ['READ_PHONE_STATE', 'READ_PHONE_NUMBERS', 'CALL_PHONE', 'ANSWER_PHONE_CALLS', 'ADD_VOICEMAIL', 'USE_SIP'],
  Sensors: ['BODY_SENSORS', 'BODY_SENSORS_BACKGROUND'],
  SMS: ['SEND_SMS', 'RECEIVE_SMS', 'READ_SMS', 'RECEIVE_WAP_PUSH', 'RECEIVE_MMS'],
  Storage: ['READ_EXTERNAL_STORAGE', 'WRITE_EXTERNAL_STORAGE', 'READ_MEDIA_IMAGES', 'READ_MEDIA_VIDEO', 'READ_MEDIA_AUDIO'],
  Bluetooth: ['BLUETOOTH_SCAN', 'BLUETOOTH_CONNECT', 'BLUETOOTH_ADVERTISE'],
  Activity: ['ACTIVITY_RECOGNITION'],
  'Nearby Devices': ['NEARBY_WIFI_DEVICES'],
  Notifications: ['POST_NOTIFICATIONS'],
};

/* ── Enforcement Actions ──────────────────────────────────────────────────── */
export const ENFORCEMENT_ACTIONS = ['no_op', 'alert', 'rate_limit', 'revoke'] as const;
export type EnforcementAction = (typeof ENFORCEMENT_ACTIONS)[number];

export const ACTION_COSTS: Record<EnforcementAction, number> = {
  no_op: 0,
  alert: 0.2,
  rate_limit: 0.5,
  revoke: 1.0,
};

export const ACTION_COLORS: Record<EnforcementAction, string> = {
  no_op: 'var(--text-tertiary)',
  alert: 'var(--accent-warning)',
  rate_limit: 'var(--accent-risk)',
  revoke: 'var(--accent-enforce)',
};

export const ACTION_ICONS: Record<EnforcementAction, string> = {
  no_op: '⏸',
  alert: '⚠',
  rate_limit: '🔻',
  revoke: '⚡',
};

/* ── Monitoring Actions ───────────────────────────────────────────────────── */
export const MONITORING_ACTIONS = ['idle', 'sample'] as const;

/* ── Risk Analysis Actions ────────────────────────────────────────────────── */
export const RISK_ACTIONS = ['defer', 'analyse'] as const;

/* ── Agent Colors ─────────────────────────────────────────────────────────── */
export const AGENT_COLORS = {
  monitoring: 'var(--accent-monitor)',
  risk: 'var(--accent-risk)',
  enforcement: 'var(--accent-enforce)',
} as const;

/* ── Paper Seeds ──────────────────────────────────────────────────────────── */
export const PAPER_SEEDS = [7, 42, 123, 777, 2024] as const;

/* ── Default Hyperparameters ──────────────────────────────────────────────── */
export const DEFAULTS = {
  numBenign: 50,
  numMalicious: 10,
  maxSteps: 200,
  epsSafe: 0.025,
  emaAlpha: 0.3,
  riskThreshold: 0.5,
  lambda1: 10.0,
  lambda2: 0.1,
  lambda3: 1.0,
  gamma: 0.99,
  gaeLambda: 0.95,
  epsClip: 0.2,
  entropyCoef: 0.01,
} as const;

/* ── Google Play Categories ───────────────────────────────────────────────── */
export const APP_CATEGORIES = [
  'Art & Design', 'Auto & Vehicles', 'Beauty', 'Books & Reference',
  'Business', 'Comics', 'Communication', 'Dating', 'Education',
  'Entertainment', 'Events', 'Finance', 'Food & Drink',
  'Health & Fitness', 'House & Home', 'Libraries & Demo',
  'Lifestyle', 'Maps & Navigation', 'Medical', 'Music & Audio',
  'News & Magazines', 'Parenting', 'Personalization', 'Photography',
  'Productivity', 'Shopping', 'Social', 'Sports', 'Tools',
  'Travel & Local', 'Video Players & Editors', 'Weather', 'Game',
] as const;

/* ── API Base URL ─────────────────────────────────────────────────────────── */
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8001';
export const WS_BASE = process.env.NEXT_PUBLIC_WS_BASE || 'ws://localhost:8001';

/* Shown when the Python inference backend is unreachable. */
export const BACKEND_HINT =
  'Cannot reach the TrustGuard backend on localhost:8001. Start it with:  python backend/main.py';
