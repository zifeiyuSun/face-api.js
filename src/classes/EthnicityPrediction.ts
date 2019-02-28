import { Prediction } from './Prediction';

// other: "like Hispanic, Latino, Middle Eastern"
export type Ethnicity = 'white' | 'black' | 'asian' | 'indian' | 'other'

export const ETHNICITIES = ['white', 'black', 'asian', 'indian', 'other'] as Ethnicity[]

export type IEthnicityPrediction = {
  white: number,
  black: number,
  asian: number,
  indian: number,
  other: number
}

export class EthnicityPrediction extends Prediction implements IEthnicityPrediction {

  public static fromProbabilities(probabilities: number[] | Float32Array): EthnicityPrediction {
    if (probabilities.length !== 5) {
      throw new Error(`EthnicityPrediction.fromProbabilities - expected probabilities.length to be 5, have: ${probabilities.length}`)
    }

    const [white, black, asian, indian, other] = probabilities as [number, number, number, number, number]
    return new EthnicityPrediction({ white, black, asian, indian, other })
  }

  private _white: number
  private _black: number
  private _asian: number
  private _indian: number
  private _other: number

  constructor(probabilities: IEthnicityPrediction) {
    super(ETHNICITIES)

    const { white, black, asian, indian, other } = probabilities
    this._white = white
    this._black = black
    this._asian = asian
    this._indian = indian
    this._other = other

    ETHNICITIES.forEach(e => super.retrieveProbability(e))
  }

  public get white(): number { return this._white }
  public get black(): number { return this._black }
  public get asian(): number { return this._asian }
  public get indian(): number { return this._indian }
  public get other(): number { return this._other }
}