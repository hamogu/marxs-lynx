from datetime import datetime

from marxs.base import MarxsElement
from . import version


class TagVersion(MarxsElement):

    def __init__(self, **kwargs):
        self.origin = kwargs.pop('origin', 'unknown')
        self.creator = kwargs.pop('creator', 'MARXS')

    def __call__(self, photons, *args, **kwargs):
        photons.meta['LYNXVER'] = (version.version, 'marxs-lynx code version')
        photons.meta['LYNXGIT'] = (version.githash, 'Git hash of marxs-lynx code')
        photons.meta['LYNXTIM'] = (version.timestamp, 'Commit time')
        photons.meta['ORIGIN'] = (self.origin, 'Institution where file was created')
        photons.meta['CREATOR'] = (self.creator, 'Person or program creating the file')
        photons.meta['DATE'] = datetime.now().isoformat()[:10]
        photons.meta['SATELLIT'] = 'LYNX'

        return photons

tagversion = TagVersion()
