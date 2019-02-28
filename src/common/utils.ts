export function descending<T>(propertyGetter: (obj: T) => number) {
  return (obj1: T, obj2: T) => propertyGetter(obj2) - propertyGetter(obj1)
}