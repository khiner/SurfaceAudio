// SPDX-License-Identifier: GPL-3.0-or-later
// Compile against the unmodified SDT revision documented in docs/Sdt.md.
#include "SDTCommon.h"
#include "SDTControl.h"
#include "SDTInteractors.h"
#include "SDTResonators.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

extern unsigned int seed;

static SDTResonator *Body(int inertial) {
    const double frequencies[] = {230, 690, 1430}, decays[] = {0.3, 0.15, 0.08};
    const double masses[] = {0.4, 0.25, 0.2}, gains[] = {0.9, 0.4, 0.25}, outputs[] = {0.3, 0.8, 0.2};
    int count = inertial ? 1 : 3;
    SDTResonator *body = SDTResonator_new(count, 2);
    for (int i = 0; i < count; ++i) {
        SDTResonator_setFrequency(body, i, inertial ? 0 : frequencies[i]);
        SDTResonator_setDecay(body, i, inertial ? 0 : decays[i]);
        SDTResonator_setWeight(body, i, inertial ? 0.03 : masses[i]);
        SDTResonator_setGain(body, 0, i, inertial ? 1 : gains[i]);
        SDTResonator_setGain(body, 1, i, inertial ? 1 : outputs[i]);
    }
    SDTResonator_setFragmentSize(body, 1);
    SDTResonator_setActiveModes(body, count);
    SDTResonator_setPosition(body, 0, 0);
    SDTResonator_setVelocity(body, 0, 0);
    return body;
}

int main(int argc, char **argv) {
    if (argc != 2) return 1;
    FILE *output = fopen(argv[1], "wb");
    if (!output) return 2;
    SDT_setSampleRate(48000);
    for (int friction = 0; friction < 2; ++friction) {
        SDTResonator *body0 = Body(1), *body1 = Body(0);
        SDTInteractor *contact = friction ? SDTFriction_new() : SDTImpact_new();
        SDTInteractor_setFirstResonator(contact, body0);
        SDTInteractor_setSecondResonator(contact, body1);
        if (friction) {
            SDTFriction_setNormalForce(contact, 0.7);
            SDTFriction_setDissipation(contact, 0.015);
            SDTFriction_setViscosity(contact, 0.01);
            SDTFriction_setNoisiness(contact, 0.0005);
            seed = 42;
        } else {
            SDTImpact_setStiffness(contact, 1000000);
            SDTImpact_setDissipation(contact, 0.1);
            SDTImpact_setShape(contact, 1.5);
        }
        SDTResonator_setVelocity(body0, 0, friction ? -0.05 : -0.5);
        for (int frame = 0; frame < 4096; ++frame) {
            const double external = friction ? -0.03 : 0;
            SDTResonator_applyForce(body0, 0, external);
            double force = SDTInteractor_computeForce(contact);
            SDTResonator_applyForce(body0, 0, force);
            SDTResonator_applyForce(body1, 0, -force);
            SDTResonator_dsp(body0);
            SDTResonator_dsp(body1);
            double values[] = {force, SDTResonator_getPosition(body0, 0), SDTResonator_getPosition(body1, 0), SDTResonator_getVelocity(body0, 0), SDTResonator_getVelocity(body1, 0), SDTResonator_getPosition(body1, 1)};
            if (fwrite(values, sizeof(double), 6, output) != 6) return 3;
        }
        if (friction) SDTFriction_free(contact);
        else SDTImpact_free(contact);
        SDTResonator_free(body0);
        SDTResonator_free(body1);
    }
    SDTRolling *rolling = SDTRolling_new();
    SDTRolling_setGrain(rolling, 0.07);
    SDTRolling_setDepth(rolling, 2);
    SDTRolling_setMass(rolling, 0.04);
    SDTRolling_setVelocity(rolling, 1.3);
    SDTScraping *scraping = SDTScraping_new();
    SDTScraping_setGrain(scraping, 0.07);
    SDTScraping_setForce(scraping, 0.8);
    SDTScraping_setVelocity(scraping, 1.3);
    for (int frame = 0; frame < 4096; ++frame) {
        const double surface = sin(frame * 0.13) + 0.3 * cos(frame * 0.71);
        double values[] = {SDTRolling_dsp(rolling, surface), SDTScraping_dsp(scraping, surface)};
        if (fwrite(values, sizeof(double), 2, output) != 2) return 3;
    }
    SDTRolling_free(rolling);
    SDTScraping_free(scraping);
    return fclose(output) != 0;
}
