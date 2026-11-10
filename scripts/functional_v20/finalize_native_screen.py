"""Record native confirmed quality failure, preserve all other uncertainty."""
from .common import EXP,ROOT,read,write,sha,stamp
from .native_review import record
from .finalize_review import finalize

def main():
    evidence='Whole960x640 input and result directly viewed. Milky Way, stars and horizon are brighter, but dense red/green grain and blocky colour patches spread through broad sky and foreground and obscure ground/mountain texture. Tree silhouettes remain; brightness is present with a major basic-quality failure. No denoising or recovered-detail claim.'
    for item in ('91ba127c6e699058','bd9f220eb7cf26cd'):
        record('native-development','old-qualified-checkpoints',item,'C','M',evidence)
    write(EXP/'native-detail-stop.json',dict(at=stamp(),status='confirmed_major_failure_stops_further_detail_expansion',
        rule='Both eligible checkpoints have a directly viewed native-pixel major failure and therefore cannot pass the zero-major gate regardless of unresolved cases.',
        directly_viewed_new_contact_outputs=65,exact_prior_output_reviews_reused=7,
        new_native_whole_outputs_viewed=2,additional_native_whole_inputs_viewed=1,
        other_contact_uncertainties_remain=True,unresolved_quality_is_not_automatically_acceptable=True,
        count_interpretation='Certified clean completions are conservative counts with unresolved cases, not an equally resolved comparison to previous native16/20 scores.',
        final_acceptance_opened=False,user_feedback=0,training_or_parameter_changes=False))
    finalize('native-development',['last-e20','last-e80'],'native-screen-major-stop')

if __name__=='__main__':main()
