/* Copyright 2021 Axel Huebl, Benjamin Worpitz, Jeffrey Kelling
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/math/isnan/Traits.hpp>

namespace alpaka
{
    namespace math
    {
        //! The standard library isnan, implementation covered by the general template.
        class IsnanStdLib : public concepts::Implements<ConceptMathIsnan, IsnanStdLib>
        {
        };
    } // namespace math
} // namespace alpaka
