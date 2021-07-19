#pragma once

#include <array>
#include <initializer_list>
#include <type_traits>
#include <cstdint>


namespace osn
{

namespace _detail
{
	constexpr int64_t PSIZE = 2048;
	constexpr int64_t PMASK = 2047;

	template<uint32_t _Dimensions, typename _Float = float>
	struct grad;

	template<uint32_t _Dimensions, typename _Float = float, typename _Int = int32_t>
	struct lattice_point;

	template<uint32_t _Dimensions, typename _Float = float>
	struct pregen_gradients;

	template<uint32_t _Dimensions, typename _Float = float>
	struct pregen_gradients_list;

	template<uint32_t _Dimensions, typename _Float = float, typename _Int = int32_t>
	struct pregen_lattice;

	template<size_t N, uint32_t _Dimensions, typename _Float, typename _Int>
	struct pregen_lattice_list_initializer;

	template<uint32_t _Dimensions, typename _Float, typename _Int>
	struct noise_impl;

	template<uint32_t _Dimensions, typename _ModeEnum, _ModeEnum mode>
	struct noise_mode_impl;

} // namespace _detail

enum class Mode
{
	Standard_2D,
	XBeforeY_2D,
	Classic_3D,
	XYBeforeZ_3D,
	XZBeforeY_3D,
	Classic_4D,
	XYBeforeZW_4D,
	XZBeforeYW_4D,
	XYZBeforeW_4D
};


template<uint32_t _Dimensions, Mode _Mode, typename _Float = float, typename _Int = int32_t>
class OpenSimplex2S
{
  private:
	std::array<uint16_t, _detail::PSIZE> perm;
	std::array<_detail::grad<_Dimensions, _Float>, _detail::PSIZE> permGrad;

  public:
	template<typename _SeedT = uint64_t>
	constexpr OpenSimplex2S(_SeedT seed = 0)
	{
		std::array<short, _detail::PSIZE> source;
		for (short i = 0; i < _detail::PSIZE; i++)
		{
			source[i] = i;
		}
		for (int32_t i = _detail::PSIZE - 1; i >= 0; i -= 1)
		{
			seed = seed * 6364136223846793005L + 1442695040888963407L;
			_SeedT r = (_SeedT)((seed + 31) % (i + 1));

			perm[i] = source[r];
			permGrad[i] = _detail::pregen_gradients<_Dimensions, _Float>::grads[perm[i]];
			source[r] = source[i];
		}
	}


	template<
	      typename... _F,
	      class = std::common_type<_Float, _F...>,
	      std::enable_if_t<(sizeof...(_F) == _Dimensions)>* = nullptr>
	_Float operator()(_F... vals)
	{
		return _detail::noise_mode_impl<_Dimensions, Mode, _Mode>::template eval<_Float, _Int>(
		      permGrad,
		      perm,
		      _Float(vals)...);
	}
};


} // namespace osn

#include "opensimplex2s.inl"
