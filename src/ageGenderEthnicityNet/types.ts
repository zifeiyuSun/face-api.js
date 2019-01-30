import { TfjsImageRecognitionBase } from 'tfjs-image-recognition-base';

export type NetParams = {
  fc: {
    age: TfjsImageRecognitionBase.FCParams
    gender: TfjsImageRecognitionBase.FCParams
    ethnicity: TfjsImageRecognitionBase.FCParams
  }
}

export const ethnicityLabels = {
  white: 0,
  black: 1,
  asian: 2,
  indian: 3,
  other: 4
}

// other: "like Hispanic, Latino, Middle Eastern"
export type Ethnicity = 'white' | 'black' | 'asian' | 'indian' | 'other'

export type EthnicityPrediction = {
  ethnicity: Ethnicity,
  probability: number
}

export const genderLabels = {
  female: 0,
  male: 1
}

export type Gender = 'female' | 'male'

export type GenderPrediction = {
  gender: Gender,
  probability: number
}