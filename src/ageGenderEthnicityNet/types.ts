import { TfjsImageRecognitionBase } from 'tfjs-image-recognition-base';

export type NetParams = {
  fc: {
    age: TfjsImageRecognitionBase.FCParams
    gender: TfjsImageRecognitionBase.FCParams
    ethnicity: TfjsImageRecognitionBase.FCParams
  }
}