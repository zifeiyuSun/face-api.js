import { extractWeightsFactory, ParamMapping } from 'tfjs-image-recognition-base';
import { extractFCParamsFactory } from 'tfjs-tiny-yolov2';

import { cOutAge, cOutEthnicity, cOutGender } from './const';
import { NetParams } from './types';

export function extractParams(weights: Float32Array, channelsIn: number): { params: NetParams, paramMappings: ParamMapping[] } {

  const paramMappings: ParamMapping[] = []

  const {
    extractWeights,
    getRemainingWeights
  } = extractWeightsFactory(weights)

  const extractFCParams = extractFCParamsFactory(extractWeights, paramMappings)

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