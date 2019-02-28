import { round, isValidProbablitiy } from 'tfjs-image-recognition-base';

import { descending } from '../common/utils';
import { LabelWithProbability } from './types';

export abstract class Prediction {

  private _labels: string[]

  constructor(labels: string[]) {
    this._labels = labels
  }

  public retrieveProbability(label: string): number {
    const probability = this[label]
    if (!isValidProbablitiy(probability)) {
      throw new Error(`Prediction.getProbability - invalid probability value: ${probability}, for label: ${label}`)
    }
    return probability
  }

  public getTopN(n: number = 1): LabelWithProbability[] {
    return this._labels
      .map(label => ({ label, probability: this.retrieveProbability(label) }))
      .sort(descending(e => e.probability))
      .slice(0, n)
  }

  public getTop(): LabelWithProbability {
    return this.getTopN(1)[0]
  }

  public toString(withDistance: boolean = true): string {
    const { label, probability } = this.getTop()
    return `${label}${withDistance ? ` (${round(probability)})` : ''}`
  }
}