# v0.2.2

Breaking changes:

- Renamed 2 utils functions
  - `utils.iter_structured` → `utils.iter_structure`
  - `utils.iter_structured_zip` → `utils.iter_zip_structures`
- rearranged parameters of methods:
  - `Apply.apply_on_state_tensor()`
  - `Measure.measure_on_state_tensor()`

Library improvements:

- Implemented 2 new operations for measurement
  - `DesiredMeasurement`
  - `ProjectiveMeasurement`
- Upgraded dependency `braandket` to 0.8.1

Developing improvements:

- Added GitHub workflow action to release on PyPI

# v0.2.1

Library improvements:

- Added trait `ControlledApply`

Developing improvements:

- Added GitHub workflow for building and releasing.

Bug fixes:

- Added the missing `CONT` in package `braandket_synthesis.operations`

# v0.2.0

Breaking changes:

- Changed the way of handling classical results:
  - Removed the definitions of `SE` (side effect) from `QOperation` and `Apply`
  - Added trait `Measure` to handle classical results.
- Renamed the 3 structural operation classes:
  - `RemappedOperation` → `Remapped`
  - `ControlledOperation` → `Controlled`
  - `SequentialOperation` → `Sequential`

Library improvements:

- Optimized the trait resolving:
  - Now the trait classes are iterated in a reversed order (newer first), allowing users to "override" existing trait implementation.
  - Now trait class mro is considered in trait resolving in addition to operation class mro. This guarantees the correct resolution of trait class when there are more than one trait implementation for the specified operation class.
  - Improved the error message about trait classes with incorrect type arguments.

# v0.1.0

The first ever release!

Included the following functionalities and features:

- Basic definition of `QOperation` and `QOperationTrait`
- Useful operations:
  - Numeric operations: `MatrixOperation` and `QubitsMatrixOperation`
  - Structural operations: `SequentialOperation`, `RemappedOperation` and `ControlledOperation`
  - Quantum Gates: `X`, `Y`, `Z`, `H`, `S`, `T`, `Rx`, `Ry`, `Rz`, `CNOT`, etc.
- Useful traits:
  - `Apply`
  - `ToTensor`, `ToKraus`
  - `IsUnitary`, `IsHermitian`, `IsDiagonal`

We know that, with only these functionalities and features, the library is still far from being good to use. We will continue to add more functionalities and features, making it better and better. And you need to be aware that quite a few
**breaking changes** may be introduced along the way. So, please use this library with great caution.
