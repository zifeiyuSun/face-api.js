import { FCParams } from 'tfjs-tiny-yolov2';

export type NetParams = {
  fc: {
    age: FCParams
    gender: FCParams
    ethnicity: FCParams
  }
}

