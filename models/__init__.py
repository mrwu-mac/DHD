from models.dhd import build_detector as build_detector_dhd


def build_detector(args, object_to_target):
    if args.model == 'dhd':
        return build_detector_dhd(args, object_to_target)
    else:
        raise