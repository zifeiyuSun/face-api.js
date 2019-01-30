

import { cOutAge, cOutEthnicity, cOutGender } from './const';
import { NetParams } from './types';
import { TfjsImageRecognitionBase } from 'tfjs-image-recognition-base';

export function extractParams(weights: Float32Array, channelsIn: number): { params: NetParams, paramMappings: TfjsImageRecognitionBase.ParamMapping[] } {

  const paramMappings: TfjsImageRecognitionBase.ParamMapping[] = []

  const {
    extractWeights,
    getRemainingWeights
  } = TfjsImageRecognitionBase.extractWeightsFactory(weights)

  const extractFCParams = TfjsImageRecognitionBase.extractFCParamsFactory(extractWeights, paramMappings)

  const age = extractFCParams(channelsIn, cOutAge, 'fc/age')
  const gender = extractFCParams(channelsIn, cOutGender, 'fc/gender')
  const ethnicity = extractFCParams(channelsIn, cOutEthnicity, 'fc/ethnicity')

  if (getRemainingWeights().length !== 0) {
    throw new Error(`weights remaing after extract: ${getRemainingWeights().length}`)
  }

  return {
    paramMappings,
    params: { fc: { age, gender, ethnicity } }
  }
}