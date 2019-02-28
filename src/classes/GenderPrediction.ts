import { Prediction } from './Prediction';

export type Gender = 'female' | 'male'

export const GENDERS = ['female', 'male'] as Gender[]

export interface IGenderPrediction {
  female: number
  male: number
}

export class GenderPrediction extends Prediction implements IGenderPrediction {

  public static fromProbabilities(probabilities: number[] | Float32Array): GenderPrediction {
    if (probabilities.length !== 2) {
      throw new Error(`GenderPrediction.fromProbabilities - expected probabilities.length to be 2, have: ${probabilities.length}`)
    }

    const probFemale = probabilities[0] as number
    return new GenderPrediction(probFemale)
  }

  private _probFemale: number

  constructor(probFemale: number) {
    super(GENDERS)
    this._probFemale = probFemale
    GENDERS.forEach(gender => super.retrieveProbability(gender))
  }

  public get female(): number { return this._probFemale }
  public get male(): number { return 1 - this._probFemale }
}