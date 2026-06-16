export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface TopKPrediction {
  sign: string;
  confidence: number;
}

export interface PredictionResponse {
  sign: string;
  confidence: number;
  confidence_pct: string;
  latency_ms: number;
  top_k: TopKPrediction[];
}

export interface TrainRequestBody {
  phase1_epochs: number;
  phase2_epochs: number;
  batch_size: number;
  model_name: string;
  backbone: string;
}

export interface TrainResponse {
  status: string;
  message: string;
}

export interface StreamMessage {
  sign?: string;
  confidence?: number;
  confidence_pct?: string;
  latency_ms?: number;
  error?: string;
}
