
namespace osn
{

namespace _detail
{


	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Generic implementations
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<typename _Float, typename _Int>
	inline constexpr _Int fastFloor(_Float x)
	{
		_Int xi = (_Int)x;
		return x < xi ? xi - 1 : xi;
	}

	template<uint32_t _Dimensions, typename _Float>
	struct grad
	{
		_Float v[_Dimensions];

		constexpr grad() {}

		template<typename... _F, class = std::common_type<_Float, _F...>>
		constexpr explicit grad(_F... vals)
		    : v{ _Float(vals)... }
		{
		}

		template<size_t N>
		constexpr grad(const _Float (&vals)[N])
		{
			static_assert(N == _Dimensions, "Number of provided arguments doesn't match grad's dimension");
			;
			for (size_t i = 0; i < N; ++i)
			{
				v[i] = vals[i];
			}
		}

		constexpr grad<_Dimensions, _Float>& operator/=(_Float f)
		{
			for (size_t i = 0; i < _Dimensions; ++i)
			{
				v[i] /= f;
			}
			return *this;
		}
	};

	template<uint32_t _Dimensions, typename _Float>
	constexpr grad<_Dimensions, _Float> operator/(grad<_Dimensions, _Float> g, _Float f)
	{
		g /= f;
		return g;
	}

	template<uint32_t _Dimensions, typename _Float>
	struct pregen_gradients
	{
		static constexpr pregen_gradients_list<_Dimensions, _Float> grads{};
	};

	template<uint32_t _Dimensions, typename _Float, typename _Int>
	struct pregen_lattice
	{
		static constexpr auto points{ pregen_lattice_list_initializer<0, _Dimensions, _Float, _Int>::init() };
	};


	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 2D specialization code
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<>
	struct noise_mode_impl<2, Mode, Mode::Standard_2D>
	{
		template<typename _Float, typename _Int>
		static constexpr _Float eval(
		      const std::array<grad<2, _Float>, PSIZE>& grads,
		      const std::array<uint16_t, PSIZE>& perm,
		      _Float x,
		      _Float y)
		{
			_Float s = _Float(0.366025403784439) * (x + y);
			_Float xs = x + s;
			_Float ys = y + s;
			return _detail::noise_impl<2, _Float, _Int>::eval(grads, perm, xs, ys);
		}
	};

	template<>
	struct noise_mode_impl<2, Mode, Mode::XBeforeY_2D>
	{
		template<typename _Float, typename _Int>
		static constexpr _Float eval(
		      const std::array<grad<2, _Float>, PSIZE>& grads,
		      const std::array<uint16_t, PSIZE>& perm,
		      _Float x,
		      _Float y)
		{
			_Float xx = x * 0.7071067811865476;
			_Float yy = y * 1.224744871380249;
			return _detail::noise_impl<2, _Float, _Int>::eval(grads, perm, yy + xx, yy - xx);
		}
	};


	template<typename _Float, typename _Int>
	struct lattice_point<2, _Float, _Int>
	{
		_Int xsv, ysv;
		_Float dx, dy;

		constexpr static _Float d_multiplicand = _Float(-0.211324865405187);
		constexpr lattice_point() = default;
		constexpr lattice_point(const lattice_point<2, _Float, _Int>&) = default;
		constexpr lattice_point(lattice_point<2, _Float, _Int>&&) = default;

		constexpr lattice_point(_Int x, _Int y)
		    : xsv(x)
		    , ysv(y)
		    , dx(-x - _Float(x + y) * d_multiplicand)
		    , dy(-y - _Float(x + y) * d_multiplicand)
		{
		}
	};

	template<typename _Float>
	struct pregen_gradients_list<2, _Float>
	{
		typedef grad<2, _Float> grad_t;
		static inline constexpr grad_t grad_div(const grad_t& g) { return g / _Float(0.05481866495625118); }

		static constexpr grad_t grads[] = { grad_div(grad_t{ 0.130526192220052, 0.99144486137381 }),
			                                grad_div(grad_t{ 0.38268343236509, 0.923879532511287 }),
			                                grad_div(grad_t{ 0.608761429008721, 0.793353340291235 }),
			                                grad_div(grad_t{ 0.793353340291235, 0.608761429008721 }),
			                                grad_div(grad_t{ 0.923879532511287, 0.38268343236509 }),
			                                grad_div(grad_t{ 0.99144486137381, 0.130526192220051 }),
			                                grad_div(grad_t{ 0.99144486137381, -0.130526192220051 }),
			                                grad_div(grad_t{ 0.923879532511287, -0.38268343236509 }),
			                                grad_div(grad_t{ 0.793353340291235, -0.60876142900872 }),
			                                grad_div(grad_t{ 0.608761429008721, -0.793353340291235 }),
			                                grad_div(grad_t{ 0.38268343236509, -0.923879532511287 }),
			                                grad_div(grad_t{ 0.130526192220052, -0.99144486137381 }),
			                                grad_div(grad_t{ -0.130526192220052, -0.99144486137381 }),
			                                grad_div(grad_t{ -0.38268343236509, -0.923879532511287 }),
			                                grad_div(grad_t{ -0.608761429008721, -0.793353340291235 }),
			                                grad_div(grad_t{ -0.793353340291235, -0.608761429008721 }),
			                                grad_div(grad_t{ -0.923879532511287, -0.38268343236509 }),
			                                grad_div(grad_t{ -0.99144486137381, -0.130526192220052 }),
			                                grad_div(grad_t{ -0.99144486137381, 0.130526192220051 }),
			                                grad_div(grad_t{ -0.923879532511287, 0.38268343236509 }),
			                                grad_div(grad_t{ -0.793353340291235, 0.608761429008721 }),
			                                grad_div(grad_t{ -0.608761429008721, 0.793353340291235 }),
			                                grad_div(grad_t{ -0.38268343236509, 0.923879532511287 }),
			                                grad_div(grad_t{ -0.130526192220052, 0.99144486137381 }) };

		static constexpr size_t n_grads = sizeof(grads) / sizeof(grad_t);

		constexpr pregen_gradients_list() = default;

		constexpr const grad_t operator[](size_t idx) const { return grads[idx % n_grads]; }
	};

	template<size_t _N, typename _Float, typename _Int>
	struct pregen_lattice_list_initializer<_N, 2, _Float, _Int>
	{
		typedef lattice_point<2, _Float, _Int> lattice_point_t;

		template<typename... _F>
		static constexpr auto init(_F... values)
		{
			_Int i1 = 0, j1 = 0, i2 = 0, j2 = 0;
			if ((_N & 1) == 0)
			{
				if ((_N & 2) == 0)
				{
					i1 = -1;
					j1 = 0;
				}
				else
				{
					i1 = 1;
					j1 = 0;
				}
				if ((_N & 4) == 0)
				{
					i2 = 0;
					j2 = -1;
				}
				else
				{
					i2 = 0;
					j2 = 1;
				}
			}
			else
			{
				if ((_N & 2) != 0)
				{
					i1 = 2;
					j1 = 1;
				}
				else
				{
					i1 = 0;
					j1 = 1;
				}
				if ((_N & 4) != 0)
				{
					i2 = 1;
					j2 = 2;
				}
				else
				{
					i2 = 1;
					j2 = 0;
				}
			}
			lattice_point_t a(0, 0);
			lattice_point_t b(1, 1);
			lattice_point_t c(i1, j1);
			lattice_point_t d(i2, j2);

			return pregen_lattice_list_initializer<_N + 1, 2, _Float, _Int>::init(values..., a, b, c, d);
		}
	};

	template<typename _Float, typename _Int>
	struct pregen_lattice_list_initializer<8, 2, _Float, _Int>
	{
		typedef lattice_point<2, _Float, _Int> lattice_point_t;

		template<typename... _F>
		static constexpr auto init(_F... values)
		{
			return std::array<lattice_point_t, 8 * 4>{ values... };
		}
	};


	template<typename _Float, typename _Int>
	struct noise_impl<2, _Float, _Int>
	{
		static constexpr _Float eval(
		      const std::array<grad<2, _Float>, PSIZE>& grads,
		      const std::array<uint16_t, PSIZE>& perm,
		      _Float xs,
		      _Float ys)
		{
			_Float value = 0;

			// Get base points and offsets
			_Int xsb = fastFloor<_Float, _Int>(xs);
			_Int ysb = fastFloor<_Float, _Int>(ys);
			_Float xsi = xs - xsb, ysi = ys - ysb;

			// Index to point list
			_Int a = std::min((_Int)(xsi + ysi), _Int(1));
			_Int index = (a << 2)
			             | (std::min(_Int(xsi - ysi / _Float(2) + _Float(1) - _Float(a) / _Float(2)), _Int(1)) << 3)
			             | (std::min(_Int(ysi - xsi / _Float(2) + _Float(1) - _Float(a) / _Float(2)), _Int(1)) << 4);

			_Float ssi = (xsi + ysi) * _Float(-0.211324865405187);
			_Float xi = xsi + ssi, yi = ysi + ssi;

			// Point contributions
			for (uint32_t i = 0; i < 4; i += 1)
			{
				lattice_point<2, _Float, _Int> c = pregen_lattice<2, _Float, _Int>::points[index + i];

				_Float dx = xi + c.dx, dy = yi + c.dy;
				_Float attn = _Float(2) / _Float(3) - dx * dx - dy * dy;
				if (attn <= 0)
					continue;

				_Int pxm = (xsb + c.xsv) & PMASK, pym = (ysb + c.ysv) & PMASK;
				grad<2, _Float> g = grads[perm[pxm] ^ pym];
				_Float extrapolation = g.v[0] * dx + g.v[1] * dy;

				attn *= attn;
				value += attn * attn * extrapolation;
			}

			return value;
		}
	};


	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 3D specialization code
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<>
	struct noise_mode_impl<3, Mode, Mode::Classic_3D>
	{
		template<typename _Float, typename _Int>
		static constexpr _Float eval(
		      const std::array<grad<3, _Float>, PSIZE>& grads,
		      const std::array<uint16_t, PSIZE>& perm,
		      _Float x,
		      _Float y,
		      _Float z)
		{
			// Re-orient the cubic lattices via rotation, to produce the expected look on cardinal planar slices.
			// If texturing objects that don't tend to have cardinal plane faces, you could even remove this.
			// Orthonormal rotation. Not a skew transform.
			_Float r = (_Float(2) / _Float(3)) * (x + y + z);
			_Float xr = r - x;
			_Float yr = r - y;
			_Float zr = r - z;

			// Evaluate both lattices to form a BCC lattice.
			return _detail::noise_impl<3, _Float, _Int>::eval(grads, perm, xr, yr, zr);
		}
	};

	template<>
	struct noise_mode_impl<3, Mode, Mode::XYBeforeZ_3D>
	{
		template<typename _Float, typename _Int>
		static constexpr _Float eval(
		      const std::array<grad<3, _Float>, PSIZE>& grads,
		      const std::array<uint16_t, PSIZE>& perm,
		      _Float x,
		      _Float y,
		      _Float z)
		{
			// Re-orient the cubic lattices without skewing, to make X and Y triangular like 2D.
			// Orthonormal rotation. Not a skew transform.
			_Float xy = x + y;
			_Float s2 = xy * _Float(-0.211324865405187);
			_Float zz = z * _Float(0.577350269189626);
			_Float xr = x + s2 - zz, yr = y + s2 - zz;
			_Float zr = xy * _Float(0.577350269189626) + zz;

			// Evaluate both lattices to form a BCC lattice.
			return _detail::noise_impl<3, _Float, _Int>::eval(grads, perm, xr, yr, zr);
		}
	};

	template<>
	struct noise_mode_impl<3, Mode, Mode::XZBeforeY_3D>
	{
		template<typename _Float, typename _Int>
		static constexpr _Float eval(
		      const std::array<grad<3, _Float>, PSIZE>& grads,
		      const std::array<uint16_t, PSIZE>& perm,
		      _Float x,
		      _Float y,
		      _Float z)
		{
			// Re-orient the cubic lattices without skewing, to make X and Z triangular like 2D.
			// Orthonormal rotation. Not a skew transform.
			_Float xz = x + z;
			_Float s2 = xz * _Float(-0.211324865405187);
			_Float yy = y * _Float(0.577350269189626);
			_Float xr = x + s2 - yy;
			_Float yr = xz * _Float(0.577350269189626) + yy;
			_Float zr = z + s2 - yy;

			// Evaluate both lattices to form a BCC lattice.
			return _detail::noise_impl<3, _Float, _Int>::eval(grads, perm, xr, yr, zr);
		}
	};


	template<typename _Float, typename _Int>
	struct lattice_point<3, _Float, _Int>
	{
		_Int xrv, yrv, zrv;
		_Float dxr, dyr, dzr;

		constexpr lattice_point() = default;
		constexpr lattice_point(const lattice_point<3, _Float, _Int>&) = default;
		constexpr lattice_point(lattice_point<3, _Float, _Int>&&) = default;

		constexpr lattice_point(_Int x, _Int y, _Int z, _Int lattice)
		    : xrv(x + lattice * (PSIZE / 2))
		    , yrv(y + lattice * (PSIZE / 2))
		    , zrv(z + lattice * (PSIZE / 2))
		    , dxr(-x + lattice * _Float(0.5))
		    , dyr(-y + lattice * _Float(0.5))
		    , dzr(-z + lattice * _Float(0.5))
		{
		}
	};

	template<typename _Float>
	struct pregen_gradients_list<3, _Float>
	{
		typedef grad<3, _Float> grad_t;

		static inline constexpr grad_t grad_div(const grad_t& g) { return g / _Float(0.2781926117527186); }
		static constexpr grad_t grads[] = { grad_div(grad_t{ -2.22474487139, -2.22474487139, -1.0 }),
			                                grad_div(grad_t{ -2.22474487139, -2.22474487139, 1.0 }),
			                                grad_div(grad_t{ -3.0862664687972017, -1.1721513422464978, 0.0 }),
			                                grad_div(grad_t{ -1.1721513422464978, -3.0862664687972017, 0.0 }),
			                                grad_div(grad_t{ -2.22474487139, -1.0, -2.22474487139 }),
			                                grad_div(grad_t{ -2.22474487139, 1.0, -2.22474487139 }),
			                                grad_div(grad_t{ -1.1721513422464978, 0.0, -3.0862664687972017 }),
			                                grad_div(grad_t{ -3.0862664687972017, 0.0, -1.1721513422464978 }),
			                                grad_div(grad_t{ -2.22474487139, -1.0, 2.22474487139 }),
			                                grad_div(grad_t{ -2.22474487139, 1.0, 2.22474487139 }),
			                                grad_div(grad_t{ -3.0862664687972017, 0.0, 1.1721513422464978 }),
			                                grad_div(grad_t{ -1.1721513422464978, 0.0, 3.0862664687972017 }),
			                                grad_div(grad_t{ -2.22474487139, 2.22474487139, -1.0 }),
			                                grad_div(grad_t{ -2.22474487139, 2.22474487139, 1.0 }),
			                                grad_div(grad_t{ -1.1721513422464978, 3.0862664687972017, 0.0 }),
			                                grad_div(grad_t{ -3.0862664687972017, 1.1721513422464978, 0.0 }),
			                                grad_div(grad_t{ -1.0, -2.22474487139, -2.22474487139 }),
			                                grad_div(grad_t{ 1.0, -2.22474487139, -2.22474487139 }),
			                                grad_div(grad_t{ 0.0, -3.0862664687972017, -1.1721513422464978 }),
			                                grad_div(grad_t{ 0.0, -1.1721513422464978, -3.0862664687972017 }),
			                                grad_div(grad_t{ -1.0, -2.22474487139, 2.22474487139 }),
			                                grad_div(grad_t{ 1.0, -2.22474487139, 2.22474487139 }),
			                                grad_div(grad_t{ 0.0, -1.1721513422464978, 3.0862664687972017 }),
			                                grad_div(grad_t{ 0.0, -3.0862664687972017, 1.1721513422464978 }),
			                                grad_div(grad_t{ -1.0, 2.22474487139, -2.22474487139 }),
			                                grad_div(grad_t{ 1.0, 2.22474487139, -2.22474487139 }),
			                                grad_div(grad_t{ 0.0, 1.1721513422464978, -3.0862664687972017 }),
			                                grad_div(grad_t{ 0.0, 3.0862664687972017, -1.1721513422464978 }),
			                                grad_div(grad_t{ -1.0, 2.22474487139, 2.22474487139 }),
			                                grad_div(grad_t{ 1.0, 2.22474487139, 2.22474487139 }),
			                                grad_div(grad_t{ 0.0, 3.0862664687972017, 1.1721513422464978 }),
			                                grad_div(grad_t{ 0.0, 1.1721513422464978, 3.0862664687972017 }),
			                                grad_div(grad_t{ 2.22474487139, -2.22474487139, -1.0 }),
			                                grad_div(grad_t{ 2.22474487139, -2.22474487139, 1.0 }),
			                                grad_div(grad_t{ 1.1721513422464978, -3.0862664687972017, 0.0 }),
			                                grad_div(grad_t{ 3.0862664687972017, -1.1721513422464978, 0.0 }),
			                                grad_div(grad_t{ 2.22474487139, -1.0, -2.22474487139 }),
			                                grad_div(grad_t{ 2.22474487139, 1.0, -2.22474487139 }),
			                                grad_div(grad_t{ 3.0862664687972017, 0.0, -1.1721513422464978 }),
			                                grad_div(grad_t{ 1.1721513422464978, 0.0, -3.0862664687972017 }),
			                                grad_div(grad_t{ 2.22474487139, -1.0, 2.22474487139 }),
			                                grad_div(grad_t{ 2.22474487139, 1.0, 2.22474487139 }),
			                                grad_div(grad_t{ 1.1721513422464978, 0.0, 3.0862664687972017 }),
			                                grad_div(grad_t{ 3.0862664687972017, 0.0, 1.1721513422464978 }),
			                                grad_div(grad_t{ 2.22474487139, 2.22474487139, -1.0 }),
			                                grad_div(grad_t{ 2.22474487139, 2.22474487139, 1.0 }),
			                                grad_div(grad_t{ 3.0862664687972017, 1.1721513422464978, 0.0 }),
			                                grad_div(grad_t{ 1.1721513422464978, 3.0862664687972017, 0.0 }) };

		static constexpr size_t n_grads = sizeof(grads) / sizeof(grad_t);

		constexpr pregen_gradients_list() = default;

		constexpr const grad_t operator[](size_t idx) const { return grads[idx % n_grads]; }
	};


	template<size_t _N, typename _Float, typename _Int>
	struct pregen_lattice_list_initializer<_N, 3, _Float, _Int>
	{
		typedef lattice_point<3, _Float, _Int> lattice_point_t;

		static constexpr auto init()
		{
			return pregen_lattice_list_initializer<_N + 1, 3, _Float, _Int>::
			      template initr<>(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
		}

		template<typename... _F, class = std::common_type<_F...>>
		static constexpr auto initr(
		      _F... values0,
		      int,
		      _F... values1,
		      int,
		      _F... values2,
		      int,
		      _F... values3,
		      int,
		      _F... values4,
		      int,
		      _F... values5,
		      int,
		      _F... values6,
		      int,
		      _F... values7,
		      int,
		      _F... values8,
		      int,
		      _F... values9,
		      int,
		      _F... valuesA,
		      int,
		      _F... valuesB,
		      int,
		      _F... valuesC,
		      int,
		      _F... valuesD)
		{
			if constexpr (_N >= 8)
			{
				return std::array<lattice_point_t, 8 * 14>{ values0..., values1..., values2..., values3..., values4...,
					                                        values5..., values6..., values7..., values8..., values9...,
					                                        valuesA..., valuesB..., valuesC..., valuesD... };
			}
			else
			{
				_Int i1 = 0, j1 = 0, k1 = 0, i2 = 0, j2 = 0, k2 = 0;

				i1 = (_N >> 0) & 1;
				j1 = (_N >> 1) & 1;
				k1 = (_N >> 2) & 1;
				i2 = i1 ^ 1;
				j2 = j1 ^ 1;
				k2 = k1 ^ 1;

				lattice_point_t c0(i1, j1, k1, 0);
				lattice_point_t c1(i1 + i2, j1 + j2, k1 + k2, 1);
				lattice_point_t c2(i1 ^ 1, j1, k1, 0);
				lattice_point_t c3(i1, j1 ^ 1, k1 ^ 1, 0);
				lattice_point_t c4(i1 + (i2 ^ 1), j1 + j2, k1 + k2, 1);
				lattice_point_t c5(i1 + i2, j1 + (j2 ^ 1), k1 + (k2 ^ 1), 1);
				lattice_point_t c6(i1, j1 ^ 1, k1, 0);
				lattice_point_t c7(i1 ^ 1, j1, k1 ^ 1, 0);
				lattice_point_t c8(i1 + i2, j1 + (j2 ^ 1), k1 + k2, 1);
				lattice_point_t c9(i1 + (i2 ^ 1), j1 + j2, k1 + (k2 ^ 1), 1);
				lattice_point_t cA(i1, j1, k1 ^ 1, 0);
				lattice_point_t cB(i1 ^ 1, j1 ^ 1, k1, 0);
				lattice_point_t cC(i1 + i2, j1 + j2, k1 + (k2 ^ 1), 1);
				lattice_point_t cD(i1 + (i2 ^ 1), j1 + (j2 ^ 1), k1 + k2, 1);

				return pregen_lattice_list_initializer<_N + 1, 3, _Float, _Int>::template initr<_F..., lattice_point_t>(
				      values0...,
				      c0,
				      0,
				      values1...,
				      c1,
				      1,
				      values2...,
				      c2,
				      2,
				      values3...,
				      c3,
				      3,
				      values4...,
				      c4,
				      4,
				      values5...,
				      c5,
				      5,
				      values6...,
				      c6,
				      6,
				      values7...,
				      c7,
				      7,
				      values8...,
				      c8,
				      8,
				      values9...,
				      c9,
				      9,
				      valuesA...,
				      cA,
				      10,
				      valuesB...,
				      cB,
				      11,
				      valuesC...,
				      cC,
				      12,
				      valuesD...,
				      cD);
			}
		}
	};


	template<typename _Float, typename _Int>
	struct noise_impl<3, _Float, _Int>
	{
		static constexpr std::array<uint8_t, 14> NextLatticeIndexBlockFailure{ 1, 2,   3,   4,   5,   6,   7,
			                                                                   8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xff };
		static constexpr std::array<uint8_t, 14> NextLatticeIndexBlockSuccess{ 1, 2,   5,   4,   6,   6,    9,
			                                                                   8, 0xA, 0xA, 0xD, 0xC, 0xff, 0xff };

		static constexpr _Float eval(
		      const std::array<grad<3, _Float>, PSIZE>& grads,
		      const std::array<uint16_t, PSIZE>& perm,
		      _Float xr,
		      _Float yr,
		      _Float zr)
		{
			// Get base and offsets inside cube of first lattice.
			_Int xrb = fastFloor<_Float, _Int>(xr);
			_Int yrb = fastFloor<_Float, _Int>(yr);
			_Int zrb = fastFloor<_Float, _Int>(zr);
			_Float xri = xr - xrb, yri = yr - yrb, zri = zr - zrb;

			// Identify which octant of the cube we're in. This determines which cell
			// in the other cubic lattice we're in, and also narrows down one point on each.
			_Int xht = (_Int)(xri + 0.5);
			_Int yht = (_Int)(yri + 0.5);
			_Int zht = (_Int)(zri + 0.5);
			_Int index = (xht << 0) | (yht << 1) | (zht << 2);

			// Point contributions
			_Float value = 0;

			_Int block = 0;

			while (block != 0xff)
			{
				lattice_point<3, _Float, _Int> c = pregen_lattice<3, _Float, _Int>::points[index + block * 8];
				_Float dxr = xri + c.dxr;
				_Float dyr = yri + c.dyr;
				_Float dzr = zri + c.dzr;
				_Float attn = _Float(0.75) - dxr * dxr - dyr * dyr - dzr * dzr;
				if (attn < 0)
				{
					block = NextLatticeIndexBlockFailure[block];
				}
				else
				{
					_Int pxm = (xrb + c.xrv) & PMASK;
					_Int pym = (yrb + c.yrv) & PMASK;
					_Int pzm = (zrb + c.zrv) & PMASK;
					grad<3, _Float> grad = grads[perm[perm[pxm] ^ pym] ^ pzm];
					_Float extrapolation = grad.v[0] * dxr + grad.v[1] * dyr + grad.v[2] * dzr;

					attn *= attn;
					value += attn * attn * extrapolation;
					block = NextLatticeIndexBlockSuccess[block];
				}
			}
			return value;
		}
	};


	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 4D specialization code
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<>
	struct noise_mode_impl<4, Mode, Mode::Classic_4D>
	{
		template<typename _Float, typename _Int>
		static constexpr _Float eval(
		      const std::array<grad<4, _Float>, PSIZE>& grads,
		      const std::array<uint16_t, PSIZE>& perm,
		      _Float x,
		      _Float y,
		      _Float z,
		      _Float w)
		{
			// Get points for A4 lattice
			_Float s = _Float(0.309016994374947) * (x + y + z + w);
			_Float xs = x + s;
			_Float ys = y + s;
			_Float zs = z + s;
			_Float ws = w + s;

			return _detail::noise_impl<4, _Float, _Int>::eval(grads, perm, xs, ys, zs, ws);
		}
	};

	template<>
	struct noise_mode_impl<4, Mode, Mode::XYBeforeZW_4D>
	{
		template<typename _Float, typename _Int>
		static constexpr _Float eval(
		      const std::array<grad<4, _Float>, PSIZE>& grads,
		      const std::array<uint16_t, PSIZE>& perm,
		      _Float x,
		      _Float y,
		      _Float z,
		      _Float w)
		{
			_Float s2 = (x + y) * _Float(-0.28522513987434876941) + (z + w) * _Float(0.83897065470611435718);
			_Float t2 = (z + w) * _Float(0.21939749883706435719) + (x + y) * _Float(-0.48214856493302476942);
			_Float xs = x + s2;
			_Float ys = y + s2;
			_Float zs = z + t2;
			_Float ws = w + t2;

			return _detail::noise_impl<4, _Float, _Int>::eval(grads, perm, xs, ys, zs, ws);
		}
	};

	template<>
	struct noise_mode_impl<4, Mode, Mode::XZBeforeYW_4D>
	{
		template<typename _Float, typename _Int>
		static constexpr _Float eval(
		      const std::array<grad<4, _Float>, PSIZE>& grads,
		      const std::array<uint16_t, PSIZE>& perm,
		      _Float x,
		      _Float y,
		      _Float z,
		      _Float w)
		{
			_Float s2 = (x + z) * _Float(-0.28522513987434876941) + (y + w) * _Float(0.83897065470611435718);
			_Float t2 = (y + w) * _Float(0.21939749883706435719) + (x + z) * _Float(-0.48214856493302476942);
			_Float xs = x + s2;
			_Float ys = y + t2;
			_Float zs = z + s2;
			_Float ws = w + t2;

			return _detail::noise_impl<4, _Float, _Int>::eval(grads, perm, xs, ys, zs, ws);
		}
	};

	template<>
	struct noise_mode_impl<4, Mode, Mode::XYZBeforeW_4D>
	{
		template<typename _Float, typename _Int>
		static constexpr _Float eval(
		      const std::array<grad<4, _Float>, PSIZE>& grads,
		      const std::array<uint16_t, PSIZE>& perm,
		      _Float x,
		      _Float y,
		      _Float z,
		      _Float w)
		{
			_Float xyz = x + y + z;
			_Float ww = w * _Float(1.118033988749894);
			_Float s2 = xyz * _Float(-0.16666666666666666) + ww;
			_Float xs = x + s2;
			_Float ys = y + s2;
			_Float zs = z + s2;
			_Float ws = _Float(-0.5) * xyz + ww;

			return _detail::noise_impl<4, _Float, _Int>::eval(grads, perm, xs, ys, zs, ws);
		}
	};


	template<typename _Float, typename _Int>
	struct lattice_point<4, _Float, _Int>
	{
		_Int xsv = 0, ysv = 0, zsv = 0, wsv = 0;
		_Float dx = 0, dy = 0, dz = 0, dw = 0;

		constexpr static _Float d_multiplicand = _Float(-0.138196601125011);
		constexpr lattice_point() = default;
		constexpr lattice_point(const lattice_point<4, _Float, _Int>&) = default;
		constexpr lattice_point(lattice_point<4, _Float, _Int>&&) = default;

		constexpr lattice_point(_Int x, _Int y, _Int z, _Int w)
		    : xsv(x)
		    , ysv(y)
		    , zsv(z)
		    , wsv(w)
		    , dx(-x - (x + y + z + w) * d_multiplicand)
		    , dy(-y - (x + y + z + w) * d_multiplicand)
		    , dz(-z - (x + y + z + w) * d_multiplicand)
		    , dw(-w - (x + y + z + w) * d_multiplicand)
		{
		}
	};


	template<typename _Float>
	struct pregen_gradients_list<4, _Float>
	{
		typedef grad<4, _Float> grad_t;
		static inline constexpr grad_t grad_div(const grad_t& g) { return g / _Float(0.11127401889945551); }

		static constexpr grad_t grads[] = {
			grad_div(grad_t{ -0.753341017856078, -0.37968289875261624, -0.37968289875261624, -0.37968289875261624 }),
			grad_div(grad_t{ -0.7821684431180708, -0.4321472685365301, -0.4321472685365301, 0.12128480194602098 }),
			grad_div(grad_t{ -0.7821684431180708, -0.4321472685365301, 0.12128480194602098, -0.4321472685365301 }),
			grad_div(grad_t{ -0.7821684431180708, 0.12128480194602098, -0.4321472685365301, -0.4321472685365301 }),
			grad_div(grad_t{ -0.8586508742123365, -0.508629699630796, 0.044802370851755174, 0.044802370851755174 }),
			grad_div(grad_t{ -0.8586508742123365, 0.044802370851755174, -0.508629699630796, 0.044802370851755174 }),
			grad_div(grad_t{ -0.8586508742123365, 0.044802370851755174, 0.044802370851755174, -0.508629699630796 }),
			grad_div(grad_t{ -0.9982828964265062, -0.03381941603233842, -0.03381941603233842, -0.03381941603233842 }),
			grad_div(grad_t{ -0.37968289875261624, -0.753341017856078, -0.37968289875261624, -0.37968289875261624 }),
			grad_div(grad_t{ -0.4321472685365301, -0.7821684431180708, -0.4321472685365301, 0.12128480194602098 }),
			grad_div(grad_t{ -0.4321472685365301, -0.7821684431180708, 0.12128480194602098, -0.4321472685365301 }),
			grad_div(grad_t{ 0.12128480194602098, -0.7821684431180708, -0.4321472685365301, -0.4321472685365301 }),
			grad_div(grad_t{ -0.508629699630796, -0.8586508742123365, 0.044802370851755174, 0.044802370851755174 }),
			grad_div(grad_t{ 0.044802370851755174, -0.8586508742123365, -0.508629699630796, 0.044802370851755174 }),
			grad_div(grad_t{ 0.044802370851755174, -0.8586508742123365, 0.044802370851755174, -0.508629699630796 }),
			grad_div(grad_t{ -0.03381941603233842, -0.9982828964265062, -0.03381941603233842, -0.03381941603233842 }),
			grad_div(grad_t{ -0.37968289875261624, -0.37968289875261624, -0.753341017856078, -0.37968289875261624 }),
			grad_div(grad_t{ -0.4321472685365301, -0.4321472685365301, -0.7821684431180708, 0.12128480194602098 }),
			grad_div(grad_t{ -0.4321472685365301, 0.12128480194602098, -0.7821684431180708, -0.4321472685365301 }),
			grad_div(grad_t{ 0.12128480194602098, -0.4321472685365301, -0.7821684431180708, -0.4321472685365301 }),
			grad_div(grad_t{ -0.508629699630796, 0.044802370851755174, -0.8586508742123365, 0.044802370851755174 }),
			grad_div(grad_t{ 0.044802370851755174, -0.508629699630796, -0.8586508742123365, 0.044802370851755174 }),
			grad_div(grad_t{ 0.044802370851755174, 0.044802370851755174, -0.8586508742123365, -0.508629699630796 }),
			grad_div(grad_t{ -0.03381941603233842, -0.03381941603233842, -0.9982828964265062, -0.03381941603233842 }),
			grad_div(grad_t{ -0.37968289875261624, -0.37968289875261624, -0.37968289875261624, -0.753341017856078 }),
			grad_div(grad_t{ -0.4321472685365301, -0.4321472685365301, 0.12128480194602098, -0.7821684431180708 }),
			grad_div(grad_t{ -0.4321472685365301, 0.12128480194602098, -0.4321472685365301, -0.7821684431180708 }),
			grad_div(grad_t{ 0.12128480194602098, -0.4321472685365301, -0.4321472685365301, -0.7821684431180708 }),
			grad_div(grad_t{ -0.508629699630796, 0.044802370851755174, 0.044802370851755174, -0.8586508742123365 }),
			grad_div(grad_t{ 0.044802370851755174, -0.508629699630796, 0.044802370851755174, -0.8586508742123365 }),
			grad_div(grad_t{ 0.044802370851755174, 0.044802370851755174, -0.508629699630796, -0.8586508742123365 }),
			grad_div(grad_t{ -0.03381941603233842, -0.03381941603233842, -0.03381941603233842, -0.9982828964265062 }),
			grad_div(grad_t{ -0.6740059517812944, -0.3239847771997537, -0.3239847771997537, 0.5794684678643381 }),
			grad_div(grad_t{ -0.7504883828755602, -0.4004672082940195, 0.15296486218853164, 0.5029860367700724 }),
			grad_div(grad_t{ -0.7504883828755602, 0.15296486218853164, -0.4004672082940195, 0.5029860367700724 }),
			grad_div(grad_t{ -0.8828161875373585, 0.08164729285680945, 0.08164729285680945, 0.4553054119602712 }),
			grad_div(grad_t{ -0.4553054119602712, -0.08164729285680945, -0.08164729285680945, 0.8828161875373585 }),
			grad_div(grad_t{ -0.5029860367700724, -0.15296486218853164, 0.4004672082940195, 0.7504883828755602 }),
			grad_div(grad_t{ -0.5029860367700724, 0.4004672082940195, -0.15296486218853164, 0.7504883828755602 }),
			grad_div(grad_t{ -0.5794684678643381, 0.3239847771997537, 0.3239847771997537, 0.6740059517812944 }),
			grad_div(grad_t{ -0.3239847771997537, -0.6740059517812944, -0.3239847771997537, 0.5794684678643381 }),
			grad_div(grad_t{ -0.4004672082940195, -0.7504883828755602, 0.15296486218853164, 0.5029860367700724 }),
			grad_div(grad_t{ 0.15296486218853164, -0.7504883828755602, -0.4004672082940195, 0.5029860367700724 }),
			grad_div(grad_t{ 0.08164729285680945, -0.8828161875373585, 0.08164729285680945, 0.4553054119602712 }),
			grad_div(grad_t{ -0.08164729285680945, -0.4553054119602712, -0.08164729285680945, 0.8828161875373585 }),
			grad_div(grad_t{ -0.15296486218853164, -0.5029860367700724, 0.4004672082940195, 0.7504883828755602 }),
			grad_div(grad_t{ 0.4004672082940195, -0.5029860367700724, -0.15296486218853164, 0.7504883828755602 }),
			grad_div(grad_t{ 0.3239847771997537, -0.5794684678643381, 0.3239847771997537, 0.6740059517812944 }),
			grad_div(grad_t{ -0.3239847771997537, -0.3239847771997537, -0.6740059517812944, 0.5794684678643381 }),
			grad_div(grad_t{ -0.4004672082940195, 0.15296486218853164, -0.7504883828755602, 0.5029860367700724 }),
			grad_div(grad_t{ 0.15296486218853164, -0.4004672082940195, -0.7504883828755602, 0.5029860367700724 }),
			grad_div(grad_t{ 0.08164729285680945, 0.08164729285680945, -0.8828161875373585, 0.4553054119602712 }),
			grad_div(grad_t{ -0.08164729285680945, -0.08164729285680945, -0.4553054119602712, 0.8828161875373585 }),
			grad_div(grad_t{ -0.15296486218853164, 0.4004672082940195, -0.5029860367700724, 0.7504883828755602 }),
			grad_div(grad_t{ 0.4004672082940195, -0.15296486218853164, -0.5029860367700724, 0.7504883828755602 }),
			grad_div(grad_t{ 0.3239847771997537, 0.3239847771997537, -0.5794684678643381, 0.6740059517812944 }),
			grad_div(grad_t{ -0.6740059517812944, -0.3239847771997537, 0.5794684678643381, -0.3239847771997537 }),
			grad_div(grad_t{ -0.7504883828755602, -0.4004672082940195, 0.5029860367700724, 0.15296486218853164 }),
			grad_div(grad_t{ -0.7504883828755602, 0.15296486218853164, 0.5029860367700724, -0.4004672082940195 }),
			grad_div(grad_t{ -0.8828161875373585, 0.08164729285680945, 0.4553054119602712, 0.08164729285680945 }),
			grad_div(grad_t{ -0.4553054119602712, -0.08164729285680945, 0.8828161875373585, -0.08164729285680945 }),
			grad_div(grad_t{ -0.5029860367700724, -0.15296486218853164, 0.7504883828755602, 0.4004672082940195 }),
			grad_div(grad_t{ -0.5029860367700724, 0.4004672082940195, 0.7504883828755602, -0.15296486218853164 }),
			grad_div(grad_t{ -0.5794684678643381, 0.3239847771997537, 0.6740059517812944, 0.3239847771997537 }),
			grad_div(grad_t{ -0.3239847771997537, -0.6740059517812944, 0.5794684678643381, -0.3239847771997537 }),
			grad_div(grad_t{ -0.4004672082940195, -0.7504883828755602, 0.5029860367700724, 0.15296486218853164 }),
			grad_div(grad_t{ 0.15296486218853164, -0.7504883828755602, 0.5029860367700724, -0.4004672082940195 }),
			grad_div(grad_t{ 0.08164729285680945, -0.8828161875373585, 0.4553054119602712, 0.08164729285680945 }),
			grad_div(grad_t{ -0.08164729285680945, -0.4553054119602712, 0.8828161875373585, -0.08164729285680945 }),
			grad_div(grad_t{ -0.15296486218853164, -0.5029860367700724, 0.7504883828755602, 0.4004672082940195 }),
			grad_div(grad_t{ 0.4004672082940195, -0.5029860367700724, 0.7504883828755602, -0.15296486218853164 }),
			grad_div(grad_t{ 0.3239847771997537, -0.5794684678643381, 0.6740059517812944, 0.3239847771997537 }),
			grad_div(grad_t{ -0.3239847771997537, -0.3239847771997537, 0.5794684678643381, -0.6740059517812944 }),
			grad_div(grad_t{ -0.4004672082940195, 0.15296486218853164, 0.5029860367700724, -0.7504883828755602 }),
			grad_div(grad_t{ 0.15296486218853164, -0.4004672082940195, 0.5029860367700724, -0.7504883828755602 }),
			grad_div(grad_t{ 0.08164729285680945, 0.08164729285680945, 0.4553054119602712, -0.8828161875373585 }),
			grad_div(grad_t{ -0.08164729285680945, -0.08164729285680945, 0.8828161875373585, -0.4553054119602712 }),
			grad_div(grad_t{ -0.15296486218853164, 0.4004672082940195, 0.7504883828755602, -0.5029860367700724 }),
			grad_div(grad_t{ 0.4004672082940195, -0.15296486218853164, 0.7504883828755602, -0.5029860367700724 }),
			grad_div(grad_t{ 0.3239847771997537, 0.3239847771997537, 0.6740059517812944, -0.5794684678643381 }),
			grad_div(grad_t{ -0.6740059517812944, 0.5794684678643381, -0.3239847771997537, -0.3239847771997537 }),
			grad_div(grad_t{ -0.7504883828755602, 0.5029860367700724, -0.4004672082940195, 0.15296486218853164 }),
			grad_div(grad_t{ -0.7504883828755602, 0.5029860367700724, 0.15296486218853164, -0.4004672082940195 }),
			grad_div(grad_t{ -0.8828161875373585, 0.4553054119602712, 0.08164729285680945, 0.08164729285680945 }),
			grad_div(grad_t{ -0.4553054119602712, 0.8828161875373585, -0.08164729285680945, -0.08164729285680945 }),
			grad_div(grad_t{ -0.5029860367700724, 0.7504883828755602, -0.15296486218853164, 0.4004672082940195 }),
			grad_div(grad_t{ -0.5029860367700724, 0.7504883828755602, 0.4004672082940195, -0.15296486218853164 }),
			grad_div(grad_t{ -0.5794684678643381, 0.6740059517812944, 0.3239847771997537, 0.3239847771997537 }),
			grad_div(grad_t{ -0.3239847771997537, 0.5794684678643381, -0.6740059517812944, -0.3239847771997537 }),
			grad_div(grad_t{ -0.4004672082940195, 0.5029860367700724, -0.7504883828755602, 0.15296486218853164 }),
			grad_div(grad_t{ 0.15296486218853164, 0.5029860367700724, -0.7504883828755602, -0.4004672082940195 }),
			grad_div(grad_t{ 0.08164729285680945, 0.4553054119602712, -0.8828161875373585, 0.08164729285680945 }),
			grad_div(grad_t{ -0.08164729285680945, 0.8828161875373585, -0.4553054119602712, -0.08164729285680945 }),
			grad_div(grad_t{ -0.15296486218853164, 0.7504883828755602, -0.5029860367700724, 0.4004672082940195 }),
			grad_div(grad_t{ 0.4004672082940195, 0.7504883828755602, -0.5029860367700724, -0.15296486218853164 }),
			grad_div(grad_t{ 0.3239847771997537, 0.6740059517812944, -0.5794684678643381, 0.3239847771997537 }),
			grad_div(grad_t{ -0.3239847771997537, 0.5794684678643381, -0.3239847771997537, -0.6740059517812944 }),
			grad_div(grad_t{ -0.4004672082940195, 0.5029860367700724, 0.15296486218853164, -0.7504883828755602 }),
			grad_div(grad_t{ 0.15296486218853164, 0.5029860367700724, -0.4004672082940195, -0.7504883828755602 }),
			grad_div(grad_t{ 0.08164729285680945, 0.4553054119602712, 0.08164729285680945, -0.8828161875373585 }),
			grad_div(grad_t{ -0.08164729285680945, 0.8828161875373585, -0.08164729285680945, -0.4553054119602712 }),
			grad_div(grad_t{ -0.15296486218853164, 0.7504883828755602, 0.4004672082940195, -0.5029860367700724 }),
			grad_div(grad_t{ 0.4004672082940195, 0.7504883828755602, -0.15296486218853164, -0.5029860367700724 }),
			grad_div(grad_t{ 0.3239847771997537, 0.6740059517812944, 0.3239847771997537, -0.5794684678643381 }),
			grad_div(grad_t{ 0.5794684678643381, -0.6740059517812944, -0.3239847771997537, -0.3239847771997537 }),
			grad_div(grad_t{ 0.5029860367700724, -0.7504883828755602, -0.4004672082940195, 0.15296486218853164 }),
			grad_div(grad_t{ 0.5029860367700724, -0.7504883828755602, 0.15296486218853164, -0.4004672082940195 }),
			grad_div(grad_t{ 0.4553054119602712, -0.8828161875373585, 0.08164729285680945, 0.08164729285680945 }),
			grad_div(grad_t{ 0.8828161875373585, -0.4553054119602712, -0.08164729285680945, -0.08164729285680945 }),
			grad_div(grad_t{ 0.7504883828755602, -0.5029860367700724, -0.15296486218853164, 0.4004672082940195 }),
			grad_div(grad_t{ 0.7504883828755602, -0.5029860367700724, 0.4004672082940195, -0.15296486218853164 }),
			grad_div(grad_t{ 0.6740059517812944, -0.5794684678643381, 0.3239847771997537, 0.3239847771997537 }),
			grad_div(grad_t{ 0.5794684678643381, -0.3239847771997537, -0.6740059517812944, -0.3239847771997537 }),
			grad_div(grad_t{ 0.5029860367700724, -0.4004672082940195, -0.7504883828755602, 0.15296486218853164 }),
			grad_div(grad_t{ 0.5029860367700724, 0.15296486218853164, -0.7504883828755602, -0.4004672082940195 }),
			grad_div(grad_t{ 0.4553054119602712, 0.08164729285680945, -0.8828161875373585, 0.08164729285680945 }),
			grad_div(grad_t{ 0.8828161875373585, -0.08164729285680945, -0.4553054119602712, -0.08164729285680945 }),
			grad_div(grad_t{ 0.7504883828755602, -0.15296486218853164, -0.5029860367700724, 0.4004672082940195 }),
			grad_div(grad_t{ 0.7504883828755602, 0.4004672082940195, -0.5029860367700724, -0.15296486218853164 }),
			grad_div(grad_t{ 0.6740059517812944, 0.3239847771997537, -0.5794684678643381, 0.3239847771997537 }),
			grad_div(grad_t{ 0.5794684678643381, -0.3239847771997537, -0.3239847771997537, -0.6740059517812944 }),
			grad_div(grad_t{ 0.5029860367700724, -0.4004672082940195, 0.15296486218853164, -0.7504883828755602 }),
			grad_div(grad_t{ 0.5029860367700724, 0.15296486218853164, -0.4004672082940195, -0.7504883828755602 }),
			grad_div(grad_t{ 0.4553054119602712, 0.08164729285680945, 0.08164729285680945, -0.8828161875373585 }),
			grad_div(grad_t{ 0.8828161875373585, -0.08164729285680945, -0.08164729285680945, -0.4553054119602712 }),
			grad_div(grad_t{ 0.7504883828755602, -0.15296486218853164, 0.4004672082940195, -0.5029860367700724 }),
			grad_div(grad_t{ 0.7504883828755602, 0.4004672082940195, -0.15296486218853164, -0.5029860367700724 }),
			grad_div(grad_t{ 0.6740059517812944, 0.3239847771997537, 0.3239847771997537, -0.5794684678643381 }),
			grad_div(grad_t{ 0.03381941603233842, 0.03381941603233842, 0.03381941603233842, 0.9982828964265062 }),
			grad_div(grad_t{ -0.044802370851755174, -0.044802370851755174, 0.508629699630796, 0.8586508742123365 }),
			grad_div(grad_t{ -0.044802370851755174, 0.508629699630796, -0.044802370851755174, 0.8586508742123365 }),
			grad_div(grad_t{ -0.12128480194602098, 0.4321472685365301, 0.4321472685365301, 0.7821684431180708 }),
			grad_div(grad_t{ 0.508629699630796, -0.044802370851755174, -0.044802370851755174, 0.8586508742123365 }),
			grad_div(grad_t{ 0.4321472685365301, -0.12128480194602098, 0.4321472685365301, 0.7821684431180708 }),
			grad_div(grad_t{ 0.4321472685365301, 0.4321472685365301, -0.12128480194602098, 0.7821684431180708 }),
			grad_div(grad_t{ 0.37968289875261624, 0.37968289875261624, 0.37968289875261624, 0.753341017856078 }),
			grad_div(grad_t{ 0.03381941603233842, 0.03381941603233842, 0.9982828964265062, 0.03381941603233842 }),
			grad_div(grad_t{ -0.044802370851755174, 0.044802370851755174, 0.8586508742123365, 0.508629699630796 }),
			grad_div(grad_t{ -0.044802370851755174, 0.508629699630796, 0.8586508742123365, -0.044802370851755174 }),
			grad_div(grad_t{ -0.12128480194602098, 0.4321472685365301, 0.7821684431180708, 0.4321472685365301 }),
			grad_div(grad_t{ 0.508629699630796, -0.044802370851755174, 0.8586508742123365, -0.044802370851755174 }),
			grad_div(grad_t{ 0.4321472685365301, -0.12128480194602098, 0.7821684431180708, 0.4321472685365301 }),
			grad_div(grad_t{ 0.4321472685365301, 0.4321472685365301, 0.7821684431180708, -0.12128480194602098 }),
			grad_div(grad_t{ 0.37968289875261624, 0.37968289875261624, 0.753341017856078, 0.37968289875261624 }),
			grad_div(grad_t{ 0.03381941603233842, 0.9982828964265062, 0.03381941603233842, 0.03381941603233842 }),
			grad_div(grad_t{ -0.044802370851755174, 0.8586508742123365, -0.044802370851755174, 0.508629699630796 }),
			grad_div(grad_t{ -0.044802370851755174, 0.8586508742123365, 0.508629699630796, -0.044802370851755174 }),
			grad_div(grad_t{ -0.12128480194602098, 0.7821684431180708, 0.4321472685365301, 0.4321472685365301 }),
			grad_div(grad_t{ 0.508629699630796, 0.8586508742123365, -0.044802370851755174, -0.044802370851755174 }),
			grad_div(grad_t{ 0.4321472685365301, 0.7821684431180708, -0.12128480194602098, 0.4321472685365301 }),
			grad_div(grad_t{ 0.4321472685365301, 0.7821684431180708, 0.4321472685365301, -0.12128480194602098 }),
			grad_div(grad_t{ 0.37968289875261624, 0.753341017856078, 0.37968289875261624, 0.37968289875261624 }),
			grad_div(grad_t{ 0.9982828964265062, 0.03381941603233842, 0.03381941603233842, 0.03381941603233842 }),
			grad_div(grad_t{ 0.8586508742123365, -0.044802370851755174, -0.044802370851755174, 0.508629699630796 }),
			grad_div(grad_t{ 0.8586508742123365, -0.044802370851755174, 0.508629699630796, -0.044802370851755174 }),
			grad_div(grad_t{ 0.7821684431180708, -0.12128480194602098, 0.4321472685365301, 0.4321472685365301 }),
			grad_div(grad_t{ 0.8586508742123365, 0.508629699630796, -0.044802370851755174, -0.044802370851755174 }),
			grad_div(grad_t{ 0.7821684431180708, 0.4321472685365301, -0.12128480194602098, 0.4321472685365301 }),
			grad_div(grad_t{ 0.7821684431180708, 0.4321472685365301, 0.4321472685365301, -0.12128480194602098 }),
			grad_div(grad_t{ 0.753341017856078, 0.37968289875261624, 0.37968289875261624, 0.37968289875261624 })
		};

		static constexpr size_t n_grads = sizeof(grads) / sizeof(grad_t);

		constexpr pregen_gradients_list() = default;

		constexpr const grad_t operator[](size_t idx) const { return grads[idx % n_grads]; }
	};

	struct pregen_lattice_list_initializer_lookup
	{
		typedef std::pair<uint8_t, std::array<uint8_t, 20>> lookup_table_pregen_t;

		// clang-format off
		static constexpr std::array<lookup_table_pregen_t, 256> table{
			lookup_table_pregen_t{20, { 0x15, 0x45, 0x51, 0x54, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA }},
			lookup_table_pregen_t{15, { 0x15, 0x45, 0x51, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x6A, 0x95, 0x96, 0x9A, 0xA6, 0xAA }},
			lookup_table_pregen_t{16, { 0x01, 0x05, 0x11, 0x15, 0x41, 0x45, 0x51, 0x55, 0x56, 0x5A, 0x66, 0x6A, 0x96, 0x9A, 0xA6, 0xAA }},
			lookup_table_pregen_t{17, { 0x01, 0x15, 0x16, 0x45, 0x46, 0x51, 0x52, 0x55, 0x56, 0x5A, 0x66, 0x6A, 0x96, 0x9A, 0xA6, 0xAA, 0xAB }},
			lookup_table_pregen_t{15, { 0x15, 0x45, 0x54, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x69, 0x6A, 0x95, 0x99, 0x9A, 0xA9, 0xAA }},
			lookup_table_pregen_t{16, { 0x05, 0x15, 0x45, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0x95, 0x96, 0x99, 0x9A, 0xAA }},
			lookup_table_pregen_t{12, { 0x05, 0x15, 0x45, 0x55, 0x56, 0x59, 0x5A, 0x66, 0x6A, 0x96, 0x9A, 0xAA }},
			lookup_table_pregen_t{15, { 0x05, 0x15, 0x16, 0x45, 0x46, 0x55, 0x56, 0x59, 0x5A, 0x66, 0x6A, 0x96, 0x9A, 0xAA, 0xAB }},
			lookup_table_pregen_t{16, { 0x04, 0x05, 0x14, 0x15, 0x44, 0x45, 0x54, 0x55, 0x59, 0x5A, 0x69, 0x6A, 0x99, 0x9A, 0xA9, 0xAA }},
			lookup_table_pregen_t{12, { 0x05, 0x15, 0x45, 0x55, 0x56, 0x59, 0x5A, 0x69, 0x6A, 0x99, 0x9A, 0xAA }},
			lookup_table_pregen_t{10, { 0x05, 0x15, 0x45, 0x55, 0x56, 0x59, 0x5A, 0x6A, 0x9A, 0xAA }},
			lookup_table_pregen_t{14, { 0x05, 0x15, 0x16, 0x45, 0x46, 0x55, 0x56, 0x59, 0x5A, 0x5B, 0x6A, 0x9A, 0xAA, 0xAB }},
			lookup_table_pregen_t{17, { 0x04, 0x15, 0x19, 0x45, 0x49, 0x54, 0x55, 0x58, 0x59, 0x5A, 0x69, 0x6A, 0x99, 0x9A, 0xA9, 0xAA, 0xAE }},
			lookup_table_pregen_t{15, { 0x05, 0x15, 0x19, 0x45, 0x49, 0x55, 0x56, 0x59, 0x5A, 0x69, 0x6A, 0x99, 0x9A, 0xAA, 0xAE }},
			lookup_table_pregen_t{14, { 0x05, 0x15, 0x19, 0x45, 0x49, 0x55, 0x56, 0x59, 0x5A, 0x5E, 0x6A, 0x9A, 0xAA, 0xAE }},
			lookup_table_pregen_t{17, { 0x05, 0x15, 0x1A, 0x45, 0x4A, 0x55, 0x56, 0x59, 0x5A, 0x5B, 0x5E, 0x6A, 0x9A, 0xAA, 0xAB, 0xAE, 0xAF }},
			lookup_table_pregen_t{15, { 0x15, 0x51, 0x54, 0x55, 0x56, 0x59, 0x65, 0x66, 0x69, 0x6A, 0x95, 0xA5, 0xA6, 0xA9, 0xAA }},
			lookup_table_pregen_t{16, { 0x11, 0x15, 0x51, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0x95, 0x96, 0xA5, 0xA6, 0xAA }},
			lookup_table_pregen_t{12, { 0x11, 0x15, 0x51, 0x55, 0x56, 0x5A, 0x65, 0x66, 0x6A, 0x96, 0xA6, 0xAA }},
			lookup_table_pregen_t{15, { 0x11, 0x15, 0x16, 0x51, 0x52, 0x55, 0x56, 0x5A, 0x65, 0x66, 0x6A, 0x96, 0xA6, 0xAA, 0xAB }},
			lookup_table_pregen_t{16, { 0x14, 0x15, 0x54, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0x95, 0x99, 0xA5, 0xA9, 0xAA }},
			lookup_table_pregen_t{14, { 0x15, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0x95, 0x9A, 0xA6, 0xA9, 0xAA }},
			lookup_table_pregen_t{14, { 0x15, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0x96, 0x9A, 0xA6, 0xAA, 0xAB }},
			lookup_table_pregen_t{13, { 0x15, 0x16, 0x55, 0x56, 0x5A, 0x66, 0x6A, 0x6B, 0x96, 0x9A, 0xA6, 0xAA, 0xAB }},
			lookup_table_pregen_t{12, { 0x14, 0x15, 0x54, 0x55, 0x59, 0x5A, 0x65, 0x69, 0x6A, 0x99, 0xA9, 0xAA }},
			lookup_table_pregen_t{14, { 0x15, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0x99, 0x9A, 0xA9, 0xAA, 0xAE }},
			lookup_table_pregen_t{11, { 0x15, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0x9A, 0xAA }},
			lookup_table_pregen_t{12, { 0x15, 0x16, 0x55, 0x56, 0x59, 0x5A, 0x66, 0x6A, 0x6B, 0x9A, 0xAA, 0xAB }},
			lookup_table_pregen_t{15, { 0x14, 0x15, 0x19, 0x54, 0x55, 0x58, 0x59, 0x5A, 0x65, 0x69, 0x6A, 0x99, 0xA9, 0xAA, 0xAE }},
			lookup_table_pregen_t{13, { 0x15, 0x19, 0x55, 0x59, 0x5A, 0x69, 0x6A, 0x6E, 0x99, 0x9A, 0xA9, 0xAA, 0xAE }},
			lookup_table_pregen_t{12, { 0x15, 0x19, 0x55, 0x56, 0x59, 0x5A, 0x69, 0x6A, 0x6E, 0x9A, 0xAA, 0xAE }},
			lookup_table_pregen_t{14, { 0x15, 0x1A, 0x55, 0x56, 0x59, 0x5A, 0x6A, 0x6B, 0x6E, 0x9A, 0xAA, 0xAB, 0xAE, 0xAF }},
			lookup_table_pregen_t{16, { 0x10, 0x11, 0x14, 0x15, 0x50, 0x51, 0x54, 0x55, 0x65, 0x66, 0x69, 0x6A, 0xA5, 0xA6, 0xA9, 0xAA }},
			lookup_table_pregen_t{12, { 0x11, 0x15, 0x51, 0x55, 0x56, 0x65, 0x66, 0x69, 0x6A, 0xA5, 0xA6, 0xAA }},
			lookup_table_pregen_t{10, { 0x11, 0x15, 0x51, 0x55, 0x56, 0x65, 0x66, 0x6A, 0xA6, 0xAA }},
			lookup_table_pregen_t{14, { 0x11, 0x15, 0x16, 0x51, 0x52, 0x55, 0x56, 0x65, 0x66, 0x67, 0x6A, 0xA6, 0xAA, 0xAB }},
			lookup_table_pregen_t{12, { 0x14, 0x15, 0x54, 0x55, 0x59, 0x65, 0x66, 0x69, 0x6A, 0xA5, 0xA9, 0xAA }},
			lookup_table_pregen_t{14, { 0x15, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0xA5, 0xA6, 0xA9, 0xAA, 0xBA }},
			lookup_table_pregen_t{11, { 0x15, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0xA6, 0xAA }},
			lookup_table_pregen_t{12, { 0x15, 0x16, 0x55, 0x56, 0x5A, 0x65, 0x66, 0x6A, 0x6B, 0xA6, 0xAA, 0xAB }},
			lookup_table_pregen_t{10, { 0x14, 0x15, 0x54, 0x55, 0x59, 0x65, 0x69, 0x6A, 0xA9, 0xAA }},
			lookup_table_pregen_t{11, { 0x15, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0xA9, 0xAA }},
			lookup_table_pregen_t{10, { 0x15, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0xAA }},
			lookup_table_pregen_t{13, { 0x15, 0x16, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0x6B, 0xAA, 0xAB }},
			lookup_table_pregen_t{14, { 0x14, 0x15, 0x19, 0x54, 0x55, 0x58, 0x59, 0x65, 0x69, 0x6A, 0x6D, 0xA9, 0xAA, 0xAE }},
			lookup_table_pregen_t{12, { 0x15, 0x19, 0x55, 0x59, 0x5A, 0x65, 0x69, 0x6A, 0x6E, 0xA9, 0xAA, 0xAE }},
			lookup_table_pregen_t{13, { 0x15, 0x19, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0x6E, 0xAA, 0xAE }},
			lookup_table_pregen_t{15, { 0x15, 0x55, 0x56, 0x59, 0x5A, 0x66, 0x69, 0x6A, 0x6B, 0x6E, 0x9A, 0xAA, 0xAB, 0xAE, 0xAF }},
			lookup_table_pregen_t{17, { 0x10, 0x15, 0x25, 0x51, 0x54, 0x55, 0x61, 0x64, 0x65, 0x66, 0x69, 0x6A, 0xA5, 0xA6, 0xA9, 0xAA, 0xBA }},
			lookup_table_pregen_t{15, { 0x11, 0x15, 0x25, 0x51, 0x55, 0x56, 0x61, 0x65, 0x66, 0x69, 0x6A, 0xA5, 0xA6, 0xAA, 0xBA }},
			lookup_table_pregen_t{14, { 0x11, 0x15, 0x25, 0x51, 0x55, 0x56, 0x61, 0x65, 0x66, 0x6A, 0x76, 0xA6, 0xAA, 0xBA }},
			lookup_table_pregen_t{17, { 0x11, 0x15, 0x26, 0x51, 0x55, 0x56, 0x62, 0x65, 0x66, 0x67, 0x6A, 0x76, 0xA6, 0xAA, 0xAB, 0xBA, 0xBB }},
			lookup_table_pregen_t{15, { 0x14, 0x15, 0x25, 0x54, 0x55, 0x59, 0x64, 0x65, 0x66, 0x69, 0x6A, 0xA5, 0xA9, 0xAA, 0xBA }},
			lookup_table_pregen_t{13, { 0x15, 0x25, 0x55, 0x65, 0x66, 0x69, 0x6A, 0x7A, 0xA5, 0xA6, 0xA9, 0xAA, 0xBA }},
			lookup_table_pregen_t{12, { 0x15, 0x25, 0x55, 0x56, 0x65, 0x66, 0x69, 0x6A, 0x7A, 0xA6, 0xAA, 0xBA }},
			lookup_table_pregen_t{14, { 0x15, 0x26, 0x55, 0x56, 0x65, 0x66, 0x6A, 0x6B, 0x7A, 0xA6, 0xAA, 0xAB, 0xBA, 0xBB }},
			lookup_table_pregen_t{14, { 0x14, 0x15, 0x25, 0x54, 0x55, 0x59, 0x64, 0x65, 0x69, 0x6A, 0x79, 0xA9, 0xAA, 0xBA }},
			lookup_table_pregen_t{12, { 0x15, 0x25, 0x55, 0x59, 0x65, 0x66, 0x69, 0x6A, 0x7A, 0xA9, 0xAA, 0xBA }},
			lookup_table_pregen_t{13, { 0x15, 0x25, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0x7A, 0xAA, 0xBA }},
			lookup_table_pregen_t{15, { 0x15, 0x55, 0x56, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0x6B, 0x7A, 0xA6, 0xAA, 0xAB, 0xBA, 0xBB }},
			lookup_table_pregen_t{17, { 0x14, 0x15, 0x29, 0x54, 0x55, 0x59, 0x65, 0x68, 0x69, 0x6A, 0x6D, 0x79, 0xA9, 0xAA, 0xAE, 0xBA, 0xBE }},
			lookup_table_pregen_t{14, { 0x15, 0x29, 0x55, 0x59, 0x65, 0x69, 0x6A, 0x6E, 0x7A, 0xA9, 0xAA, 0xAE, 0xBA, 0xBE }},
			lookup_table_pregen_t{15, { 0x15, 0x55, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0x6E, 0x7A, 0xA9, 0xAA, 0xAE, 0xBA, 0xBE }},
			lookup_table_pregen_t{17, { 0x15, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0x6B, 0x6E, 0x7A, 0xAA, 0xAB, 0xAE, 0xBA, 0xBF }},
			lookup_table_pregen_t{15, { 0x45, 0x51, 0x54, 0x55, 0x56, 0x59, 0x65, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA }},
			lookup_table_pregen_t{16, { 0x41, 0x45, 0x51, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xAA }},
			lookup_table_pregen_t{12, { 0x41, 0x45, 0x51, 0x55, 0x56, 0x5A, 0x66, 0x95, 0x96, 0x9A, 0xA6, 0xAA }},
			lookup_table_pregen_t{15, { 0x41, 0x45, 0x46, 0x51, 0x52, 0x55, 0x56, 0x5A, 0x66, 0x95, 0x96, 0x9A, 0xA6, 0xAA, 0xAB }},
			lookup_table_pregen_t{16, { 0x44, 0x45, 0x54, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x69, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA9, 0xAA }},
			lookup_table_pregen_t{14, { 0x45, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x6A, 0x95, 0x96, 0x99, 0x9A, 0xA6, 0xA9, 0xAA }},
			lookup_table_pregen_t{14, { 0x45, 0x55, 0x56, 0x59, 0x5A, 0x66, 0x6A, 0x95, 0x96, 0x99, 0x9A, 0xA6, 0xAA, 0xAB }},
			lookup_table_pregen_t{13, { 0x45, 0x46, 0x55, 0x56, 0x5A, 0x66, 0x6A, 0x96, 0x9A, 0x9B, 0xA6, 0xAA, 0xAB }},
			lookup_table_pregen_t{12, { 0x44, 0x45, 0x54, 0x55, 0x59, 0x5A, 0x69, 0x95, 0x99, 0x9A, 0xA9, 0xAA }},
			lookup_table_pregen_t{14, { 0x45, 0x55, 0x56, 0x59, 0x5A, 0x69, 0x6A, 0x95, 0x96, 0x99, 0x9A, 0xA9, 0xAA, 0xAE }},
			lookup_table_pregen_t{11, { 0x45, 0x55, 0x56, 0x59, 0x5A, 0x6A, 0x95, 0x96, 0x99, 0x9A, 0xAA }},
			lookup_table_pregen_t{12, { 0x45, 0x46, 0x55, 0x56, 0x59, 0x5A, 0x6A, 0x96, 0x9A, 0x9B, 0xAA, 0xAB }},
			lookup_table_pregen_t{15, { 0x44, 0x45, 0x49, 0x54, 0x55, 0x58, 0x59, 0x5A, 0x69, 0x95, 0x99, 0x9A, 0xA9, 0xAA, 0xAE }},
			lookup_table_pregen_t{13, { 0x45, 0x49, 0x55, 0x59, 0x5A, 0x69, 0x6A, 0x99, 0x9A, 0x9E, 0xA9, 0xAA, 0xAE }},
			lookup_table_pregen_t{12, { 0x45, 0x49, 0x55, 0x56, 0x59, 0x5A, 0x6A, 0x99, 0x9A, 0x9E, 0xAA, 0xAE }},
			lookup_table_pregen_t{14, { 0x45, 0x4A, 0x55, 0x56, 0x59, 0x5A, 0x6A, 0x9A, 0x9B, 0x9E, 0xAA, 0xAB, 0xAE, 0xAF }},
			lookup_table_pregen_t{16, { 0x50, 0x51, 0x54, 0x55, 0x56, 0x59, 0x65, 0x66, 0x69, 0x95, 0x96, 0x99, 0xA5, 0xA6, 0xA9, 0xAA }},
			lookup_table_pregen_t{14, { 0x51, 0x55, 0x56, 0x59, 0x65, 0x66, 0x6A, 0x95, 0x96, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA }},
			lookup_table_pregen_t{14, { 0x51, 0x55, 0x56, 0x5A, 0x65, 0x66, 0x6A, 0x95, 0x96, 0x9A, 0xA5, 0xA6, 0xAA, 0xAB }},
			lookup_table_pregen_t{13, { 0x51, 0x52, 0x55, 0x56, 0x5A, 0x66, 0x6A, 0x96, 0x9A, 0xA6, 0xA7, 0xAA, 0xAB }},
			lookup_table_pregen_t{14, { 0x54, 0x55, 0x56, 0x59, 0x65, 0x69, 0x6A, 0x95, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA }},
			lookup_table_pregen_t{16, { 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA }},
			lookup_table_pregen_t{16, { 0x15, 0x45, 0x51, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x6A, 0x95, 0x96, 0x9A, 0xA6, 0xAA, 0xAB }},
			lookup_table_pregen_t{10, { 0x55, 0x56, 0x5A, 0x66, 0x6A, 0x96, 0x9A, 0xA6, 0xAA, 0xAB }},
			lookup_table_pregen_t{14, { 0x54, 0x55, 0x59, 0x5A, 0x65, 0x69, 0x6A, 0x95, 0x99, 0x9A, 0xA5, 0xA9, 0xAA, 0xAE }},
			lookup_table_pregen_t{16, { 0x15, 0x45, 0x54, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x69, 0x6A, 0x95, 0x99, 0x9A, 0xA9, 0xAA, 0xAE }},
			lookup_table_pregen_t{19, { 0x15, 0x45, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0x95, 0x96, 0x99, 0x9A, 0xA6, 0xA9, 0xAA, 0xAB, 0xAE }},
			lookup_table_pregen_t{11, { 0x55, 0x56, 0x59, 0x5A, 0x66, 0x6A, 0x96, 0x9A, 0xA6, 0xAA, 0xAB }},
			lookup_table_pregen_t{13, { 0x54, 0x55, 0x58, 0x59, 0x5A, 0x69, 0x6A, 0x99, 0x9A, 0xA9, 0xAA, 0xAD, 0xAE }},
			lookup_table_pregen_t{10, { 0x55, 0x59, 0x5A, 0x69, 0x6A, 0x99, 0x9A, 0xA9, 0xAA, 0xAE }},
			lookup_table_pregen_t{11, { 0x55, 0x56, 0x59, 0x5A, 0x69, 0x6A, 0x99, 0x9A, 0xA9, 0xAA, 0xAE }},
			lookup_table_pregen_t{10, { 0x55, 0x56, 0x59, 0x5A, 0x6A, 0x9A, 0xAA, 0xAB, 0xAE, 0xAF }},
			lookup_table_pregen_t{12, { 0x50, 0x51, 0x54, 0x55, 0x65, 0x66, 0x69, 0x95, 0xA5, 0xA6, 0xA9, 0xAA }},
			lookup_table_pregen_t{14, { 0x51, 0x55, 0x56, 0x65, 0x66, 0x69, 0x6A, 0x95, 0x96, 0xA5, 0xA6, 0xA9, 0xAA, 0xBA }},
			lookup_table_pregen_t{11, { 0x51, 0x55, 0x56, 0x65, 0x66, 0x6A, 0x95, 0x96, 0xA5, 0xA6, 0xAA }},
			lookup_table_pregen_t{12, { 0x51, 0x52, 0x55, 0x56, 0x65, 0x66, 0x6A, 0x96, 0xA6, 0xA7, 0xAA, 0xAB }},
			lookup_table_pregen_t{14, { 0x54, 0x55, 0x59, 0x65, 0x66, 0x69, 0x6A, 0x95, 0x99, 0xA5, 0xA6, 0xA9, 0xAA, 0xBA }},
			lookup_table_pregen_t{16, { 0x15, 0x51, 0x54, 0x55, 0x56, 0x59, 0x65, 0x66, 0x69, 0x6A, 0x95, 0xA5, 0xA6, 0xA9, 0xAA, 0xBA }},
			lookup_table_pregen_t{19, { 0x15, 0x51, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0x95, 0x96, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xAB, 0xBA }},
			lookup_table_pregen_t{11, { 0x55, 0x56, 0x5A, 0x65, 0x66, 0x6A, 0x96, 0x9A, 0xA6, 0xAA, 0xAB }},
			lookup_table_pregen_t{11, { 0x54, 0x55, 0x59, 0x65, 0x69, 0x6A, 0x95, 0x99, 0xA5, 0xA9, 0xAA }},
			lookup_table_pregen_t{19, { 0x15, 0x54, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0x95, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xAE, 0xBA }},
			lookup_table_pregen_t{13, { 0x15, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0x9A, 0xA6, 0xA9, 0xAA }},
			lookup_table_pregen_t{14, { 0x15, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0x96, 0x9A, 0xA6, 0xAA, 0xAB }},
			lookup_table_pregen_t{12, { 0x54, 0x55, 0x58, 0x59, 0x65, 0x69, 0x6A, 0x99, 0xA9, 0xAA, 0xAD, 0xAE }},
			lookup_table_pregen_t{11, { 0x55, 0x59, 0x5A, 0x65, 0x69, 0x6A, 0x99, 0x9A, 0xA9, 0xAA, 0xAE }},
			lookup_table_pregen_t{14, { 0x15, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0x99, 0x9A, 0xA9, 0xAA, 0xAE }},
			lookup_table_pregen_t{13, { 0x15, 0x55, 0x56, 0x59, 0x5A, 0x66, 0x69, 0x6A, 0x9A, 0xAA, 0xAB, 0xAE, 0xAF }},
			lookup_table_pregen_t{15, { 0x50, 0x51, 0x54, 0x55, 0x61, 0x64, 0x65, 0x66, 0x69, 0x95, 0xA5, 0xA6, 0xA9, 0xAA, 0xBA }},
			lookup_table_pregen_t{13, { 0x51, 0x55, 0x61, 0x65, 0x66, 0x69, 0x6A, 0xA5, 0xA6, 0xA9, 0xAA, 0xB6, 0xBA }},
			lookup_table_pregen_t{12, { 0x51, 0x55, 0x56, 0x61, 0x65, 0x66, 0x6A, 0xA5, 0xA6, 0xAA, 0xB6, 0xBA }},
			lookup_table_pregen_t{14, { 0x51, 0x55, 0x56, 0x62, 0x65, 0x66, 0x6A, 0xA6, 0xA7, 0xAA, 0xAB, 0xB6, 0xBA, 0xBB }},
			lookup_table_pregen_t{13, { 0x54, 0x55, 0x64, 0x65, 0x66, 0x69, 0x6A, 0xA5, 0xA6, 0xA9, 0xAA, 0xB9, 0xBA }},
			lookup_table_pregen_t{10, { 0x55, 0x65, 0x66, 0x69, 0x6A, 0xA5, 0xA6, 0xA9, 0xAA, 0xBA }},
			lookup_table_pregen_t{11, { 0x55, 0x56, 0x65, 0x66, 0x69, 0x6A, 0xA5, 0xA6, 0xA9, 0xAA, 0xBA }},
			lookup_table_pregen_t{10, { 0x55, 0x56, 0x65, 0x66, 0x6A, 0xA6, 0xAA, 0xAB, 0xBA, 0xBB }},
			lookup_table_pregen_t{12, { 0x54, 0x55, 0x59, 0x64, 0x65, 0x69, 0x6A, 0xA5, 0xA9, 0xAA, 0xB9, 0xBA }},
			lookup_table_pregen_t{11, { 0x55, 0x59, 0x65, 0x66, 0x69, 0x6A, 0xA5, 0xA6, 0xA9, 0xAA, 0xBA }},
			lookup_table_pregen_t{14, { 0x15, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0xA5, 0xA6, 0xA9, 0xAA, 0xBA }},
			lookup_table_pregen_t{13, { 0x15, 0x55, 0x56, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0xA6, 0xAA, 0xAB, 0xBA, 0xBB }},
			lookup_table_pregen_t{14, { 0x54, 0x55, 0x59, 0x65, 0x68, 0x69, 0x6A, 0xA9, 0xAA, 0xAD, 0xAE, 0xB9, 0xBA, 0xBE }},
			lookup_table_pregen_t{10, { 0x55, 0x59, 0x65, 0x69, 0x6A, 0xA9, 0xAA, 0xAE, 0xBA, 0xBE }},
			lookup_table_pregen_t{13, { 0x15, 0x55, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0xA9, 0xAA, 0xAE, 0xBA, 0xBE }},
			lookup_table_pregen_t{13, { 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0xAA, 0xAB, 0xAE, 0xBA, 0xBF }},
			lookup_table_pregen_t{16, { 0x40, 0x41, 0x44, 0x45, 0x50, 0x51, 0x54, 0x55, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA }},
			lookup_table_pregen_t{12, { 0x41, 0x45, 0x51, 0x55, 0x56, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xAA }},
			lookup_table_pregen_t{10, { 0x41, 0x45, 0x51, 0x55, 0x56, 0x95, 0x96, 0x9A, 0xA6, 0xAA }},
			lookup_table_pregen_t{14, { 0x41, 0x45, 0x46, 0x51, 0x52, 0x55, 0x56, 0x95, 0x96, 0x97, 0x9A, 0xA6, 0xAA, 0xAB }},
			lookup_table_pregen_t{12, { 0x44, 0x45, 0x54, 0x55, 0x59, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA9, 0xAA }},
			lookup_table_pregen_t{14, { 0x45, 0x55, 0x56, 0x59, 0x5A, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xEA }},
			lookup_table_pregen_t{11, { 0x45, 0x55, 0x56, 0x59, 0x5A, 0x95, 0x96, 0x99, 0x9A, 0xA6, 0xAA }},
			lookup_table_pregen_t{12, { 0x45, 0x46, 0x55, 0x56, 0x5A, 0x95, 0x96, 0x9A, 0x9B, 0xA6, 0xAA, 0xAB }},
			lookup_table_pregen_t{10, { 0x44, 0x45, 0x54, 0x55, 0x59, 0x95, 0x99, 0x9A, 0xA9, 0xAA }},
			lookup_table_pregen_t{11, { 0x45, 0x55, 0x56, 0x59, 0x5A, 0x95, 0x96, 0x99, 0x9A, 0xA9, 0xAA }},
			lookup_table_pregen_t{10, { 0x45, 0x55, 0x56, 0x59, 0x5A, 0x95, 0x96, 0x99, 0x9A, 0xAA }},
			lookup_table_pregen_t{13, { 0x45, 0x46, 0x55, 0x56, 0x59, 0x5A, 0x95, 0x96, 0x99, 0x9A, 0x9B, 0xAA, 0xAB }},
			lookup_table_pregen_t{14, { 0x44, 0x45, 0x49, 0x54, 0x55, 0x58, 0x59, 0x95, 0x99, 0x9A, 0x9D, 0xA9, 0xAA, 0xAE }},
			lookup_table_pregen_t{12, { 0x45, 0x49, 0x55, 0x59, 0x5A, 0x95, 0x99, 0x9A, 0x9E, 0xA9, 0xAA, 0xAE }},
			lookup_table_pregen_t{13, { 0x45, 0x49, 0x55, 0x56, 0x59, 0x5A, 0x95, 0x96, 0x99, 0x9A, 0x9E, 0xAA, 0xAE }},
			lookup_table_pregen_t{15, { 0x45, 0x55, 0x56, 0x59, 0x5A, 0x6A, 0x96, 0x99, 0x9A, 0x9B, 0x9E, 0xAA, 0xAB, 0xAE, 0xAF }},
			lookup_table_pregen_t{12, { 0x50, 0x51, 0x54, 0x55, 0x65, 0x95, 0x96, 0x99, 0xA5, 0xA6, 0xA9, 0xAA }},
			lookup_table_pregen_t{14, { 0x51, 0x55, 0x56, 0x65, 0x66, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xEA }},
			lookup_table_pregen_t{11, { 0x51, 0x55, 0x56, 0x65, 0x66, 0x95, 0x96, 0x9A, 0xA5, 0xA6, 0xAA }},
			lookup_table_pregen_t{12, { 0x51, 0x52, 0x55, 0x56, 0x66, 0x95, 0x96, 0x9A, 0xA6, 0xA7, 0xAA, 0xAB }},
			lookup_table_pregen_t{14, { 0x54, 0x55, 0x59, 0x65, 0x69, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xEA }},
			lookup_table_pregen_t{16, { 0x45, 0x51, 0x54, 0x55, 0x56, 0x59, 0x65, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xEA }},
			lookup_table_pregen_t{19, { 0x45, 0x51, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x6A, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xAB, 0xEA }},
			lookup_table_pregen_t{11, { 0x55, 0x56, 0x5A, 0x66, 0x6A, 0x95, 0x96, 0x9A, 0xA6, 0xAA, 0xAB }},
			lookup_table_pregen_t{11, { 0x54, 0x55, 0x59, 0x65, 0x69, 0x95, 0x99, 0x9A, 0xA5, 0xA9, 0xAA }},
			lookup_table_pregen_t{19, { 0x45, 0x54, 0x55, 0x56, 0x59, 0x5A, 0x65, 0x69, 0x6A, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xAE, 0xEA }},
			lookup_table_pregen_t{13, { 0x45, 0x55, 0x56, 0x59, 0x5A, 0x6A, 0x95, 0x96, 0x99, 0x9A, 0xA6, 0xA9, 0xAA }},
			lookup_table_pregen_t{14, { 0x45, 0x55, 0x56, 0x59, 0x5A, 0x66, 0x6A, 0x95, 0x96, 0x99, 0x9A, 0xA6, 0xAA, 0xAB }},
			lookup_table_pregen_t{12, { 0x54, 0x55, 0x58, 0x59, 0x69, 0x95, 0x99, 0x9A, 0xA9, 0xAA, 0xAD, 0xAE }},
			lookup_table_pregen_t{11, { 0x55, 0x59, 0x5A, 0x69, 0x6A, 0x95, 0x99, 0x9A, 0xA9, 0xAA, 0xAE }},
			lookup_table_pregen_t{14, { 0x45, 0x55, 0x56, 0x59, 0x5A, 0x69, 0x6A, 0x95, 0x96, 0x99, 0x9A, 0xA9, 0xAA, 0xAE }},
			lookup_table_pregen_t{13, { 0x45, 0x55, 0x56, 0x59, 0x5A, 0x6A, 0x96, 0x99, 0x9A, 0xAA, 0xAB, 0xAE, 0xAF }},
			lookup_table_pregen_t{10, { 0x50, 0x51, 0x54, 0x55, 0x65, 0x95, 0xA5, 0xA6, 0xA9, 0xAA }},
			lookup_table_pregen_t{11, { 0x51, 0x55, 0x56, 0x65, 0x66, 0x95, 0x96, 0xA5, 0xA6, 0xA9, 0xAA }},
			lookup_table_pregen_t{10, { 0x51, 0x55, 0x56, 0x65, 0x66, 0x95, 0x96, 0xA5, 0xA6, 0xAA }},
			lookup_table_pregen_t{13, { 0x51, 0x52, 0x55, 0x56, 0x65, 0x66, 0x95, 0x96, 0xA5, 0xA6, 0xA7, 0xAA, 0xAB }},
			lookup_table_pregen_t{11, { 0x54, 0x55, 0x59, 0x65, 0x69, 0x95, 0x99, 0xA5, 0xA6, 0xA9, 0xAA }},
			lookup_table_pregen_t{19, { 0x51, 0x54, 0x55, 0x56, 0x59, 0x65, 0x66, 0x69, 0x6A, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xBA, 0xEA }},
			lookup_table_pregen_t{13, { 0x51, 0x55, 0x56, 0x65, 0x66, 0x6A, 0x95, 0x96, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA }},
			lookup_table_pregen_t{14, { 0x51, 0x55, 0x56, 0x5A, 0x65, 0x66, 0x6A, 0x95, 0x96, 0x9A, 0xA5, 0xA6, 0xAA, 0xAB }},
			lookup_table_pregen_t{10, { 0x54, 0x55, 0x59, 0x65, 0x69, 0x95, 0x99, 0xA5, 0xA9, 0xAA }},
			lookup_table_pregen_t{13, { 0x54, 0x55, 0x59, 0x65, 0x69, 0x6A, 0x95, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA }},
			lookup_table_pregen_t{16, { 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA }},
			lookup_table_pregen_t{14, { 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x6A, 0x95, 0x96, 0x9A, 0xA6, 0xA9, 0xAA, 0xAB }},
			lookup_table_pregen_t{13, { 0x54, 0x55, 0x58, 0x59, 0x65, 0x69, 0x95, 0x99, 0xA5, 0xA9, 0xAA, 0xAD, 0xAE }},
			lookup_table_pregen_t{14, { 0x54, 0x55, 0x59, 0x5A, 0x65, 0x69, 0x6A, 0x95, 0x99, 0x9A, 0xA5, 0xA9, 0xAA, 0xAE }},
			lookup_table_pregen_t{14, { 0x55, 0x56, 0x59, 0x5A, 0x65, 0x69, 0x6A, 0x95, 0x99, 0x9A, 0xA6, 0xA9, 0xAA, 0xAE }},
			lookup_table_pregen_t{16, { 0x55, 0x56, 0x59, 0x5A, 0x66, 0x69, 0x6A, 0x96, 0x99, 0x9A, 0xA6, 0xA9, 0xAA, 0xAB, 0xAE, 0xAF }},
			lookup_table_pregen_t{14, { 0x50, 0x51, 0x54, 0x55, 0x61, 0x64, 0x65, 0x95, 0xA5, 0xA6, 0xA9, 0xAA, 0xB5, 0xBA }},
			lookup_table_pregen_t{12, { 0x51, 0x55, 0x61, 0x65, 0x66, 0x95, 0xA5, 0xA6, 0xA9, 0xAA, 0xB6, 0xBA }},
			lookup_table_pregen_t{13, { 0x51, 0x55, 0x56, 0x61, 0x65, 0x66, 0x95, 0x96, 0xA5, 0xA6, 0xAA, 0xB6, 0xBA }},
			lookup_table_pregen_t{15, { 0x51, 0x55, 0x56, 0x65, 0x66, 0x6A, 0x96, 0xA5, 0xA6, 0xA7, 0xAA, 0xAB, 0xB6, 0xBA, 0xBB }},
			lookup_table_pregen_t{12, { 0x54, 0x55, 0x64, 0x65, 0x69, 0x95, 0xA5, 0xA6, 0xA9, 0xAA, 0xB9, 0xBA }},
			lookup_table_pregen_t{11, { 0x55, 0x65, 0x66, 0x69, 0x6A, 0x95, 0xA5, 0xA6, 0xA9, 0xAA, 0xBA }},
			lookup_table_pregen_t{14, { 0x51, 0x55, 0x56, 0x65, 0x66, 0x69, 0x6A, 0x95, 0x96, 0xA5, 0xA6, 0xA9, 0xAA, 0xBA }},
			lookup_table_pregen_t{13, { 0x51, 0x55, 0x56, 0x65, 0x66, 0x6A, 0x96, 0xA5, 0xA6, 0xAA, 0xAB, 0xBA, 0xBB }},
			lookup_table_pregen_t{13, { 0x54, 0x55, 0x59, 0x64, 0x65, 0x69, 0x95, 0x99, 0xA5, 0xA9, 0xAA, 0xB9, 0xBA }},
			lookup_table_pregen_t{14, { 0x54, 0x55, 0x59, 0x65, 0x66, 0x69, 0x6A, 0x95, 0x99, 0xA5, 0xA6, 0xA9, 0xAA, 0xBA }},
			lookup_table_pregen_t{14, { 0x55, 0x56, 0x59, 0x65, 0x66, 0x69, 0x6A, 0x95, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xBA }},
			lookup_table_pregen_t{16, { 0x55, 0x56, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0x96, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xAB, 0xBA, 0xBB }},
			lookup_table_pregen_t{15, { 0x54, 0x55, 0x59, 0x65, 0x69, 0x6A, 0x99, 0xA5, 0xA9, 0xAA, 0xAD, 0xAE, 0xB9, 0xBA, 0xBE }},
			lookup_table_pregen_t{13, { 0x54, 0x55, 0x59, 0x65, 0x69, 0x6A, 0x99, 0xA5, 0xA9, 0xAA, 0xAE, 0xBA, 0xBE }},
			lookup_table_pregen_t{16, { 0x55, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xAE, 0xBA, 0xBE }},
			lookup_table_pregen_t{15, { 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0x9A, 0xA6, 0xA9, 0xAA, 0xAB, 0xAE, 0xBA }},
			lookup_table_pregen_t{17, { 0x40, 0x45, 0x51, 0x54, 0x55, 0x85, 0x91, 0x94, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xEA }},
			lookup_table_pregen_t{15, { 0x41, 0x45, 0x51, 0x55, 0x56, 0x85, 0x91, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xAA, 0xEA }},
			lookup_table_pregen_t{14, { 0x41, 0x45, 0x51, 0x55, 0x56, 0x85, 0x91, 0x95, 0x96, 0x9A, 0xA6, 0xAA, 0xD6, 0xEA }},
			lookup_table_pregen_t{17, { 0x41, 0x45, 0x51, 0x55, 0x56, 0x86, 0x92, 0x95, 0x96, 0x97, 0x9A, 0xA6, 0xAA, 0xAB, 0xD6, 0xEA, 0xEB }},
			lookup_table_pregen_t{15, { 0x44, 0x45, 0x54, 0x55, 0x59, 0x85, 0x94, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA9, 0xAA, 0xEA }},
			lookup_table_pregen_t{13, { 0x45, 0x55, 0x85, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xDA, 0xEA }},
			lookup_table_pregen_t{12, { 0x45, 0x55, 0x56, 0x85, 0x95, 0x96, 0x99, 0x9A, 0xA6, 0xAA, 0xDA, 0xEA }},
			lookup_table_pregen_t{14, { 0x45, 0x55, 0x56, 0x86, 0x95, 0x96, 0x9A, 0x9B, 0xA6, 0xAA, 0xAB, 0xDA, 0xEA, 0xEB }},
			lookup_table_pregen_t{14, { 0x44, 0x45, 0x54, 0x55, 0x59, 0x85, 0x94, 0x95, 0x99, 0x9A, 0xA9, 0xAA, 0xD9, 0xEA }},
			lookup_table_pregen_t{12, { 0x45, 0x55, 0x59, 0x85, 0x95, 0x96, 0x99, 0x9A, 0xA9, 0xAA, 0xDA, 0xEA }},
			lookup_table_pregen_t{13, { 0x45, 0x55, 0x56, 0x59, 0x5A, 0x85, 0x95, 0x96, 0x99, 0x9A, 0xAA, 0xDA, 0xEA }},
			lookup_table_pregen_t{15, { 0x45, 0x55, 0x56, 0x5A, 0x95, 0x96, 0x99, 0x9A, 0x9B, 0xA6, 0xAA, 0xAB, 0xDA, 0xEA, 0xEB }},
			lookup_table_pregen_t{17, { 0x44, 0x45, 0x54, 0x55, 0x59, 0x89, 0x95, 0x98, 0x99, 0x9A, 0x9D, 0xA9, 0xAA, 0xAE, 0xD9, 0xEA, 0xEE }},
			lookup_table_pregen_t{14, { 0x45, 0x55, 0x59, 0x89, 0x95, 0x99, 0x9A, 0x9E, 0xA9, 0xAA, 0xAE, 0xDA, 0xEA, 0xEE }},
			lookup_table_pregen_t{15, { 0x45, 0x55, 0x59, 0x5A, 0x95, 0x96, 0x99, 0x9A, 0x9E, 0xA9, 0xAA, 0xAE, 0xDA, 0xEA, 0xEE }},
			lookup_table_pregen_t{17, { 0x45, 0x55, 0x56, 0x59, 0x5A, 0x95, 0x96, 0x99, 0x9A, 0x9B, 0x9E, 0xAA, 0xAB, 0xAE, 0xDA, 0xEA, 0xEF }},
			lookup_table_pregen_t{15, { 0x50, 0x51, 0x54, 0x55, 0x65, 0x91, 0x94, 0x95, 0x96, 0x99, 0xA5, 0xA6, 0xA9, 0xAA, 0xEA }},
			lookup_table_pregen_t{13, { 0x51, 0x55, 0x91, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xE6, 0xEA }},
			lookup_table_pregen_t{12, { 0x51, 0x55, 0x56, 0x91, 0x95, 0x96, 0x9A, 0xA5, 0xA6, 0xAA, 0xE6, 0xEA }},
			lookup_table_pregen_t{14, { 0x51, 0x55, 0x56, 0x92, 0x95, 0x96, 0x9A, 0xA6, 0xA7, 0xAA, 0xAB, 0xE6, 0xEA, 0xEB }},
			lookup_table_pregen_t{13, { 0x54, 0x55, 0x94, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xE9, 0xEA }},
			lookup_table_pregen_t{10, { 0x55, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xEA }},
			lookup_table_pregen_t{11, { 0x55, 0x56, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xEA }},
			lookup_table_pregen_t{10, { 0x55, 0x56, 0x95, 0x96, 0x9A, 0xA6, 0xAA, 0xAB, 0xEA, 0xEB }},
			lookup_table_pregen_t{12, { 0x54, 0x55, 0x59, 0x94, 0x95, 0x99, 0x9A, 0xA5, 0xA9, 0xAA, 0xE9, 0xEA }},
			lookup_table_pregen_t{11, { 0x55, 0x59, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xEA }},
			lookup_table_pregen_t{14, { 0x45, 0x55, 0x56, 0x59, 0x5A, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xEA }},
			lookup_table_pregen_t{13, { 0x45, 0x55, 0x56, 0x5A, 0x95, 0x96, 0x99, 0x9A, 0xA6, 0xAA, 0xAB, 0xEA, 0xEB }},
			lookup_table_pregen_t{14, { 0x54, 0x55, 0x59, 0x95, 0x98, 0x99, 0x9A, 0xA9, 0xAA, 0xAD, 0xAE, 0xE9, 0xEA, 0xEE }},
			lookup_table_pregen_t{10, { 0x55, 0x59, 0x95, 0x99, 0x9A, 0xA9, 0xAA, 0xAE, 0xEA, 0xEE }},
			lookup_table_pregen_t{13, { 0x45, 0x55, 0x59, 0x5A, 0x95, 0x96, 0x99, 0x9A, 0xA9, 0xAA, 0xAE, 0xEA, 0xEE }},
			lookup_table_pregen_t{13, { 0x55, 0x56, 0x59, 0x5A, 0x95, 0x96, 0x99, 0x9A, 0xAA, 0xAB, 0xAE, 0xEA, 0xEF }},
			lookup_table_pregen_t{14, { 0x50, 0x51, 0x54, 0x55, 0x65, 0x91, 0x94, 0x95, 0xA5, 0xA6, 0xA9, 0xAA, 0xE5, 0xEA }},
			lookup_table_pregen_t{12, { 0x51, 0x55, 0x65, 0x91, 0x95, 0x96, 0xA5, 0xA6, 0xA9, 0xAA, 0xE6, 0xEA }},
			lookup_table_pregen_t{13, { 0x51, 0x55, 0x56, 0x65, 0x66, 0x91, 0x95, 0x96, 0xA5, 0xA6, 0xAA, 0xE6, 0xEA }},
			lookup_table_pregen_t{15, { 0x51, 0x55, 0x56, 0x66, 0x95, 0x96, 0x9A, 0xA5, 0xA6, 0xA7, 0xAA, 0xAB, 0xE6, 0xEA, 0xEB }},
			lookup_table_pregen_t{12, { 0x54, 0x55, 0x65, 0x94, 0x95, 0x99, 0xA5, 0xA6, 0xA9, 0xAA, 0xE9, 0xEA }},
			lookup_table_pregen_t{11, { 0x55, 0x65, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xEA }},
			lookup_table_pregen_t{14, { 0x51, 0x55, 0x56, 0x65, 0x66, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xEA }},
			lookup_table_pregen_t{13, { 0x51, 0x55, 0x56, 0x66, 0x95, 0x96, 0x9A, 0xA5, 0xA6, 0xAA, 0xAB, 0xEA, 0xEB }},
			lookup_table_pregen_t{13, { 0x54, 0x55, 0x59, 0x65, 0x69, 0x94, 0x95, 0x99, 0xA5, 0xA9, 0xAA, 0xE9, 0xEA }},
			lookup_table_pregen_t{14, { 0x54, 0x55, 0x59, 0x65, 0x69, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xEA }},
			lookup_table_pregen_t{14, { 0x55, 0x56, 0x59, 0x65, 0x6A, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xEA }},
			lookup_table_pregen_t{16, { 0x55, 0x56, 0x5A, 0x66, 0x6A, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xAB, 0xEA, 0xEB }},
			lookup_table_pregen_t{15, { 0x54, 0x55, 0x59, 0x69, 0x95, 0x99, 0x9A, 0xA5, 0xA9, 0xAA, 0xAD, 0xAE, 0xE9, 0xEA, 0xEE }},
			lookup_table_pregen_t{13, { 0x54, 0x55, 0x59, 0x69, 0x95, 0x99, 0x9A, 0xA5, 0xA9, 0xAA, 0xAE, 0xEA, 0xEE }},
			lookup_table_pregen_t{16, { 0x55, 0x59, 0x5A, 0x69, 0x6A, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xAE, 0xEA, 0xEE }},
			lookup_table_pregen_t{15, { 0x55, 0x56, 0x59, 0x5A, 0x6A, 0x95, 0x96, 0x99, 0x9A, 0xA6, 0xA9, 0xAA, 0xAB, 0xAE, 0xEA }},
			lookup_table_pregen_t{17, { 0x50, 0x51, 0x54, 0x55, 0x65, 0x95, 0xA1, 0xA4, 0xA5, 0xA6, 0xA9, 0xAA, 0xB5, 0xBA, 0xE5, 0xEA, 0xFA }},
			lookup_table_pregen_t{14, { 0x51, 0x55, 0x65, 0x95, 0xA1, 0xA5, 0xA6, 0xA9, 0xAA, 0xB6, 0xBA, 0xE6, 0xEA, 0xFA }},
			lookup_table_pregen_t{15, { 0x51, 0x55, 0x65, 0x66, 0x95, 0x96, 0xA5, 0xA6, 0xA9, 0xAA, 0xB6, 0xBA, 0xE6, 0xEA, 0xFA }},
			lookup_table_pregen_t{17, { 0x51, 0x55, 0x56, 0x65, 0x66, 0x95, 0x96, 0xA5, 0xA6, 0xA7, 0xAA, 0xAB, 0xB6, 0xBA, 0xE6, 0xEA, 0xFB }},
			lookup_table_pregen_t{14, { 0x54, 0x55, 0x65, 0x95, 0xA4, 0xA5, 0xA6, 0xA9, 0xAA, 0xB9, 0xBA, 0xE9, 0xEA, 0xFA }},
			lookup_table_pregen_t{10, { 0x55, 0x65, 0x95, 0xA5, 0xA6, 0xA9, 0xAA, 0xBA, 0xEA, 0xFA }},
			lookup_table_pregen_t{13, { 0x51, 0x55, 0x65, 0x66, 0x95, 0x96, 0xA5, 0xA6, 0xA9, 0xAA, 0xBA, 0xEA, 0xFA }},
			lookup_table_pregen_t{13, { 0x55, 0x56, 0x65, 0x66, 0x95, 0x96, 0xA5, 0xA6, 0xAA, 0xAB, 0xBA, 0xEA, 0xFB }},
			lookup_table_pregen_t{15, { 0x54, 0x55, 0x65, 0x69, 0x95, 0x99, 0xA5, 0xA6, 0xA9, 0xAA, 0xB9, 0xBA, 0xE9, 0xEA, 0xFA }},
			lookup_table_pregen_t{13, { 0x54, 0x55, 0x65, 0x69, 0x95, 0x99, 0xA5, 0xA6, 0xA9, 0xAA, 0xBA, 0xEA, 0xFA }},
			lookup_table_pregen_t{16, { 0x55, 0x65, 0x66, 0x69, 0x6A, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xBA, 0xEA, 0xFA }},
			lookup_table_pregen_t{15, { 0x55, 0x56, 0x65, 0x66, 0x6A, 0x95, 0x96, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xAB, 0xBA, 0xEA }},
			lookup_table_pregen_t{17, { 0x54, 0x55, 0x59, 0x65, 0x69, 0x95, 0x99, 0xA5, 0xA9, 0xAA, 0xAD, 0xAE, 0xB9, 0xBA, 0xE9, 0xEA, 0xFE }},
			lookup_table_pregen_t{13, { 0x55, 0x59, 0x65, 0x69, 0x95, 0x99, 0xA5, 0xA9, 0xAA, 0xAE, 0xBA, 0xEA, 0xFE }},
			lookup_table_pregen_t{15, { 0x55, 0x59, 0x65, 0x69, 0x6A, 0x95, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xAE, 0xBA, 0xEA }},
			lookup_table_pregen_t{20, { 0x55, 0x56, 0x59, 0x5A, 0x65, 0x66, 0x69, 0x6A, 0x95, 0x96, 0x99, 0x9A, 0xA5, 0xA6, 0xA9, 0xAA, 0xAB, 0xAE, 0xBA, 0xEA }}
		};
		// clang-format on
	};

	template<size_t _N, typename _Float, typename _Int>
	struct pregen_lattice_list_row_initializer
	{
		typedef lattice_point<4, _Float, _Int> lattice_point_t;

		typedef std::pair<uint8_t, std::array<lattice_point_t, 20>> lattice_lookup_row_t;

		template<typename... _F>
		static constexpr auto init(_F... values)
		{
			if constexpr (sizeof...(values) >= 20)
			{
				return lattice_lookup_row_t{ pregen_lattice_list_initializer_lookup::table[_N].first,
					                         std::array<lattice_point_t, 20>{ values... } };
			}
			else if constexpr (sizeof...(values) < pregen_lattice_list_initializer_lookup::table[_N].first)
			{
				// TODO: calculate the lattice point position
				uint8_t idx = pregen_lattice_list_initializer_lookup::table[_N].second[sizeof...(values)];
				_Int cx = ((idx >> 0) & 3) - 1;
				_Int cy = ((idx >> 2) & 3) - 1;
				_Int cz = ((idx >> 4) & 3) - 1;
				_Int cw = ((idx >> 6) & 3) - 1;
				return pregen_lattice_list_row_initializer<_N, _Float, _Int>::init(
				      values...,
				      lattice_point_t{ cx, cy, cz, cw });
			}
			else
			{
				return pregen_lattice_list_row_initializer<_N, _Float, _Int>::init(values..., lattice_point_t{});
			}
		}
	};

	template<size_t _N, typename _Float, typename _Int>
	struct pregen_lattice_list_initializer<_N, 4, _Float, _Int>
	{
		typedef lattice_point<4, _Float, _Int> lattice_point_t;

		template<typename... _F>
		static constexpr auto init(_F... values)
		{
			if constexpr (_N >= 256)
			{
				return std::array<pregen_lattice_list_row_initializer<_N, _Float, _Int>::lattice_lookup_row_t, 256>{
					values...
				};
			}
			else
			{
				return pregen_lattice_list_initializer<_N + 1, 4, _Float, _Int>::init(
				      values...,
				      pregen_lattice_list_row_initializer<_N, _Float, _Int>::init());
			}
		}
	};

	template<typename _Float, typename _Int>
	struct noise_impl<4, _Float, _Int>
	{
		static constexpr _Float eval(
		      const std::array<grad<4, _Float>, PSIZE>& grads,
		      const std::array<uint16_t, PSIZE>& perm,
		      _Float xs,
		      _Float ys,
		      _Float zs,
		      _Float ws)
		{
			_Float value = 0;

			// Get base points and offsets
			_Int xsb = fastFloor<_Float, _Int>(xs);
			_Int ysb = fastFloor<_Float, _Int>(ys);
			_Int zsb = fastFloor<_Float, _Int>(zs);
			_Int wsb = fastFloor<_Float, _Int>(ws);

			_Float xsi = xs - xsb;
			_Float ysi = ys - ysb;
			_Float zsi = zs - zsb;
			_Float wsi = ws - wsb;

			// Unskewed offsets
			_Float ssi = (xsi + ysi + zsi + wsi) * _Float(-0.138196601125011);
			_Float xi = xsi + ssi, yi = ysi + ssi, zi = zsi + ssi, wi = wsi + ssi;

			_Int index = ((fastFloor<_Float, _Int>(xs * 4) & 3) << 0) | ((fastFloor<_Float, _Int>(ys * 4) & 3) << 2)
			             | ((fastFloor<_Float, _Int>(zs * 4) & 3) << 4) | ((fastFloor<_Float, _Int>(ws * 4) & 3) << 6);

			// Point contributions
			for (size_t i = 0; i < pregen_lattice<4, _Float, _Int>::points[index].first; i += 1)
			{
				lattice_point<4, _Float, _Int> c = pregen_lattice<4, _Float, _Int>::points[index].second[i];

				_Float dx = xi + c.dx;
				_Float dy = yi + c.dy;
				_Float dz = zi + c.dz;
				_Float dw = wi + c.dw;

				_Float attn = _Float(0.8) - dx * dx - dy * dy - dz * dz - dw * dw;
				if (attn > 0)
				{
					_Int pxm = (xsb + c.xsv) & PMASK;
					_Int pym = (ysb + c.ysv) & PMASK;
					_Int pzm = (zsb + c.zsv) & PMASK;
					_Int pwm = (wsb + c.wsv) & PMASK;

					grad<4, _Float> grad = grads[perm[perm[perm[pxm] ^ pym] ^ pzm] ^ pwm];
					_Float extrapolation = grad.v[0] * dx + grad.v[1] * dy + grad.v[2] * dz + grad.v[3] * dw;

					attn *= attn;
					value += attn * attn * extrapolation;
				}
			}

			return value;
		}
	};


} // namespace _detail

} // namespace osn
