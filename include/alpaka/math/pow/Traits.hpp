/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Concepts.hpp>
#include <alpaka/core/Unused.hpp>

#include <cmath>

namespace alpaka
{
    namespace math
    {
        struct ConceptMathPow
        {
        };

        namespace traits
        {
            //! The pow trait.
            template<typename T, typename TBase, typename TExp, typename TSfinae = void>
            struct Pow
            {
                ALPAKA_FN_HOST_ACC auto operator()(T const& ctx, TBase const& base, TExp const& exp)
                {
                    alpaka::ignore_unused(ctx);
                    // This is an ADL call. If you get a compile error here then your type is not supported by the
                    // backend and we could not find pow(base, exp) in the namespace of your type.
                    using std::pow;
                    return pow(base, exp);
                }
            };
        } // namespace traits

        //! Computes the value of base raised to the power exp.
        //!
        //! Valid real arguments for base are non-negative. For other values
        //! the result may depend on the backend and compilation options, will
        //! likely be NaN.
        //!
        //! \tparam T The type of the object specializing Pow.
        //! \tparam TBase The base type.
        //! \tparam TExp The exponent type.
        //! \param pow_ctx The object specializing Pow.
        //! \param base The base.
        //! \param exp The exponent.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, typename TBase, typename TExp>
        ALPAKA_FN_HOST_ACC auto pow(T const& pow_ctx, TBase const& base, TExp const& exp)
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptMathPow, T>;
            return traits::Pow<ImplementationBase, TBase, TExp>{}(pow_ctx, base, exp);
        }
    } // namespace math
} // namespace alpaka
