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
        struct ConceptMathExp
        {
        };

        namespace traits
        {
            //! The exp trait.
            template<typename T, typename TArg, typename TSfinae = void>
            struct Exp
            {
                ALPAKA_FN_HOST_ACC auto operator()(T const& ctx, TArg const& arg)
                {
                    alpaka::ignore_unused(ctx);
                    // This is an ADL call. If you get a compile error here then your type is not supported by the
                    // backend and we could not find exp(TArg) in the namespace of your type.
                    using std::exp;
                    return exp(arg);
                }
            };
        } // namespace traits

        //! Computes the e (Euler's number, 2.7182818) raised to the given power arg.
        //!
        //! \tparam T The type of the object specializing Exp.
        //! \tparam TArg The arg type.
        //! \param exp_ctx The object specializing Exp.
        //! \param arg The arg.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename T, typename TArg>
        ALPAKA_FN_HOST_ACC auto exp(T const& exp_ctx, TArg const& arg)
        {
            using ImplementationBase = concepts::ImplementationBase<ConceptMathExp, T>;
            return traits::Exp<ImplementationBase, TArg>{}(exp_ctx, arg);
        }
    } // namespace math
} // namespace alpaka
